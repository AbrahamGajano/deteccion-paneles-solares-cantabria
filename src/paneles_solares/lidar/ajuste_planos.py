"""Ajuste geométrico y decisión conservadora para inclinaciones LiDAR."""

import numpy as np
from shapely import intersects_xy

CRITERIOS = {
    "porcentaje_minimo": 75.,
    "min_puntos": 10,
    "dispersion_minima": 0.25,
    "inclinacion_maxima": 75.,
    "angulo_maximo": 5.,
    "separacion_maxima": 0.30,
    "estabilidad_maxima": 3.,
    "anillo_maximo": 8.,
    "max_planos": 6,
    "distancia_vecinos": 1.,
    "inclinacion_predeterminada": 30.,
}


def ajustar_plano(puntos: np.ndarray) -> tuple:
    """Obtiene el plano de menor error perpendicular mediante SVD.

    Params:
        puntos: Matriz de al menos tres puntos XYZ que no estén alineados.

    Returns:
        tuple: Centro del conjunto y normal unitaria orientada hacia arriba.
    """

    if len(puntos) < 3:
        raise ValueError("Se necesitan al menos tres puntos para ajustar un plano.")

    centro = puntos.mean(axis=0)
    _, valores, vectores = np.linalg.svd(puntos - centro, full_matrices=False)

    if valores[1] < 1e-8:
        raise ValueError("Los puntos están alineados; no definen un plano único.")

    normal = vectores[-1]

    if normal[2] < 0:
        normal = -normal

    return centro, normal


def distancias_al_plano(puntos: np.ndarray, centro: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Distancias XYZ en metros con signo hacia la normal unitaria del plano."""

    return (puntos - centro) @ normal


def buscar_plano(
    puntos: np.ndarray, umbral: float = 0.10,
    iteraciones: int = 500, semilla: int = 42,
) -> dict:
    """Busca un plano por consenso RANSAC y lo reajusta con sus puntos compatibles.

    Params:
        puntos: Matriz XYZ de los puntos dentro de la detección.
        umbral: Distancia perpendicular máxima al plano, en metros.
        iteraciones: Número de combinaciones de tres puntos que se probarán.
        semilla: Valor para repetir la misma selección aleatoria.

    Returns:
        dict: Centro, normal, distancias absolutas y con signo, y puntos compatibles.
    """

    if len(puntos) < 3:
        raise ValueError("Hay menos de tres puntos dentro de la detección.")

    if umbral <= 0 or iteraciones < 1:
        raise ValueError("El umbral y las iteraciones deben ser positivos.")

    origen = puntos.mean(axis=0)
    locales = puntos - origen
    generador = np.random.default_rng(semilla)
    mejor_mascara = None
    mejor_cantidad = 0
    mejor_error = float("inf")

    for _ in range(iteraciones):
        indices = generador.choice(len(locales), size=3, replace=False)
        p1, p2, p3 = locales[indices]
        vector_1 = p2 - p1
        vector_2 = p3 - p1
        normal = np.cross(vector_1, vector_2)
        longitud = np.linalg.norm(normal)
        escala = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)

        if escala == 0 or longitud <= 1e-6 * escala:
            continue

        normal = normal / longitud
        distancias = np.abs((locales - p1) @ normal)
        compatibles = distancias <= umbral
        cantidad = int(compatibles.sum())
        error = float(distancias[compatibles].mean())

        if cantidad > mejor_cantidad or (
            cantidad == mejor_cantidad and error < mejor_error
        ):
            mejor_mascara = compatibles
            mejor_cantidad = cantidad
            mejor_error = error

    if mejor_mascara is None:
        raise ValueError("No se encontró una combinación que defina un plano.")

    centro, normal = ajustar_plano(locales[mejor_mascara])
    distancias_con_signo = distancias_al_plano(locales, centro, normal)
    distancias = np.abs(distancias_con_signo)
    compatibles = distancias <= umbral

    if compatibles.sum() < 3:
        raise ValueError("El plano reajustado no conserva tres puntos compatibles.")

    return {
        "centro": centro + origen,
        "normal": normal,
        "distancias": distancias,
        "distancias_con_signo": distancias_con_signo,
        "compatibles": compatibles,
    }


def calcular_inclinacion(normal: np.ndarray) -> float:
    """Ángulo del plano respecto a la horizontal, en grados."""

    return float(np.degrees(np.arctan2(np.hypot(normal[0], normal[1]), abs(normal[2]))))


def angulo_entre_planos(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    """Ángulo de 0 a 90 grados entre normales unitarias, incluidas vertientes opuestas."""

    coseno = np.clip(abs(normal_a @ normal_b), 0., 1.)
    return float(np.degrees(np.arccos(coseno)))


def evaluar_distribucion(puntos: np.ndarray) -> dict:
    """Mide la extensión horizontal en las dos direcciones principales.

    Params:
        puntos: Puntos XYZ compatibles con un plano.

    Returns:
        dict: Desviaciones espaciales en metros; la menor detecta apoyos estrechos.
    """

    if len(puntos) < 3:
        return {"dispersion_mayor_m": 0., "dispersion_menor_m": 0.}

    xy = puntos[:, :2] - puntos[:, :2].mean(axis=0)
    valores = np.linalg.svd(xy, compute_uv=False) / np.sqrt(len(puntos))
    return {"dispersion_mayor_m": float(valores[0]),
            "dispersion_menor_m": float(valores[1])}


def evaluar_calidad(puntos: np.ndarray, ajuste: dict, criterios: dict,
                    porcentaje_minimo: float = 0.) -> dict:
    """Comprueba cantidad, distribución e inclinación de un candidato.

    Params:
        puntos: Todos los puntos del conjunto ajustado.
        ajuste: Resultado de buscar_plano.
        criterios: Límites configurables de CRITERIOS.
        porcentaje_minimo: Apoyo exigido; cero para fragmentos exteriores.

    Returns:
        dict: Medidas y primer motivo de descarte, o None si supera los límites.
    """

    compatibles = puntos[ajuste["compatibles"]]
    medidas = evaluar_distribucion(compatibles)
    medidas["n_compatibles"] = len(compatibles)
    medidas["porcentaje_compatible"] = 100. * len(compatibles) / len(puntos)
    medidas["inclinacion_grados"] = calcular_inclinacion(ajuste["normal"])
    medidas["motivo"] = None

    for falla, motivo in [
        (len(compatibles) < criterios["min_puntos"], "pocos_inliers"),
        (medidas["porcentaje_compatible"] < porcentaje_minimo, "apoyo_interior_insuficiente"),
        (medidas["dispersion_menor_m"] < criterios["dispersion_minima"], "apoyo_casi_lineal"),
        (medidas["inclinacion_grados"] > criterios["inclinacion_maxima"], "inclinacion_fuera_de_alcance"),
    ]:
        if falla:
            medidas["motivo"] = motivo
            break

    return medidas


def evaluar_estabilidad(puntos: np.ndarray, ajuste: dict, umbral: float,
                        iteraciones: int, semilla: int, criterios: dict) -> dict:
    """Repite el ajuste con dos tolerancias y una submuestra reproducible.

    Params:
        puntos: Conjunto original sobre el que repetir RANSAC.
        ajuste: Ajuste de referencia, que no se reemplaza por otro más favorable.
        umbral: Tolerancia base en metros.
        iteraciones: Pruebas RANSAC por repetición.
        semilla: Semilla base para reproducir pruebas y submuestra.
        criterios: Límites de distribución y variación angular.

    Returns:
        dict: Variaciones máximas de orientación y posición, y resultado de la prueba.
    """

    generador = np.random.default_rng(semilla)
    indices = generador.choice(len(puntos), max(3, int(0.8 * len(puntos))), replace=False)
    pruebas = [(puntos, 0.8 * umbral), (puntos, 1.2 * umbral), (puntos[indices], umbral)]
    angulos = []
    separaciones = []

    for numero, (muestra, tolerancia) in enumerate(pruebas, start=1):
        try:
            repetido = buscar_plano(muestra, tolerancia, iteraciones, semilla + numero)
        except ValueError:
            return {"estable": False, "variacion_angular_grados": None,
                    "variacion_posicion_m": None}

        distribucion = evaluar_distribucion(muestra[repetido["compatibles"]])
        if distribucion["dispersion_menor_m"] < criterios["dispersion_minima"]:
            return {"estable": False, "variacion_angular_grados": None,
                    "variacion_posicion_m": None}

        angulos.append(angulo_entre_planos(ajuste["normal"], repetido["normal"]))
        separaciones.append(abs(float((ajuste["centro"] - repetido["centro"]) @ repetido["normal"])))

    return {
        "estable": max(angulos) <= criterios["estabilidad_maxima"] and max(separaciones) <= 2 * umbral,
        "variacion_angular_grados": max(angulos),
        "variacion_posicion_m": max(separaciones),
    }


def separar_grupos(puntos: np.ndarray, paso: float) -> list:
    """Separa parches mediante celdas XY ocupadas y sus ocho vecinas.

    Params:
        puntos: Puntos de un mismo candidato plano.
        paso: Lado de las celdas en metros; es una aproximación de conectividad.

    Returns:
        list: Arrays de índices por componente, en orden reproducible.
    """

    celdas = np.floor((puntos[:, :2] - puntos[:, :2].min(axis=0)) / paso).astype(int)
    ocupadas = {}
    for indice, celda in enumerate(celdas):
        ocupadas.setdefault(tuple(celda), []).append(indice)

    grupos = []
    while ocupadas:
        pendientes = [next(iter(ocupadas))]
        indices = []
        while pendientes:
            x, y = pendientes.pop()
            if (x, y) not in ocupadas:
                continue
            indices.extend(ocupadas.pop((x, y)))
            pendientes.extend((x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                               if (x + dx, y + dy) in ocupadas)
        grupos.append(np.array(sorted(indices)))

    return grupos


def contar_vecinos(puntos: np.ndarray, referencia: np.ndarray, distancia: float) -> int:
    """Cuenta puntos próximos en 3D a la referencia usando bloques pequeños.

    Params:
        puntos: Puntos exteriores que se quieren asociar.
        referencia: Inliers interiores del panel.
        distancia: Distancia euclídea máxima entre puntos, en metros.

    Returns:
        int: Número de puntos exteriores con algún vecino interior cercano.
    """

    cantidad = 0
    for inicio in range(0, len(puntos), 128):
        bloque = puntos[inicio:inicio + 128]
        cerca = np.zeros(len(bloque), dtype=bool)
        for posicion in range(0, len(referencia), 512):
            diferencias = bloque[:, None, :] - referencia[None, posicion:posicion + 512, :]
            cerca |= (np.sum(diferencias**2, axis=2) <= distancia**2).any(axis=1)
        cantidad += int(cerca.sum())

    return cantidad


def buscar_candidatos_exteriores(puntos: np.ndarray, umbral: float, iteraciones: int,
                                 semilla: int, criterios: dict) -> list:
    """Extrae sucesivamente planos y sus parches hasta el límite configurado.

    Params:
        puntos: Puntos de una franja exterior.
        umbral: Tolerancia RANSAC exterior en metros.
        iteraciones: Número de pruebas por ajuste.
        semilla: Semilla reproducible.
        criterios: Límite de planos, cantidad, extensión e inclinación admitidas.

    Returns:
        list: Candidatos con sus puntos, plano y medidas de calidad.
    """

    pendientes = puntos
    candidatos = []
    for numero in range(criterios["max_planos"]):
        if len(pendientes) < criterios["min_puntos"]:
            break
        try:
            ajuste = buscar_plano(pendientes, umbral, iteraciones, semilla + numero)
        except ValueError:
            break

        compatibles = pendientes[ajuste["compatibles"]]
        if len(compatibles) < criterios["min_puntos"]:
            break
        pendientes = pendientes[~ajuste["compatibles"]]
        for indices in separar_grupos(compatibles, criterios["distancia_vecinos"]):
            fragmento = compatibles[indices]
            if len(fragmento) < criterios["min_puntos"]:
                continue
            try:
                centro, normal = ajustar_plano(fragmento)
            except ValueError:
                continue
            distancias = distancias_al_plano(fragmento, centro, normal)
            plano = {"centro": centro, "normal": normal,
                     "compatibles": np.abs(distancias) <= umbral}
            calidad = evaluar_calidad(fragmento, plano, criterios)
            if calidad["motivo"] is None:
                candidatos.append({"puntos": fragmento[plano["compatibles"]],
                                   "ajuste": plano, "calidad": calidad})

    return candidatos


def buscar_cubierta(contexto: np.ndarray, poligono, dentro: np.ndarray, ajuste: dict,
                    umbral: float, anillo: float, iteraciones: int, semilla: int,
                    criterios: dict, ventana=None) -> dict:
    """Busca una superficie exterior próxima, sin elegirla por su paralelismo.

    Params:
        contexto: Todos los puntos del recorte.
        poligono: Detección original en coordenadas métricas.
        dentro: Máscara de puntos interiores.
        ajuste: Plano interior ya comprobado.
        umbral: Tolerancia del ajuste exterior.
        anillo: Anchura inicial; se duplica hasta anillo_maximo.
        iteraciones: Pruebas RANSAC por candidato.
        semilla: Semilla reproducible.
        criterios: Límites de calidad y proximidad.
        ventana: Extensión disponible del recorte, o None si se desconoce.

    Returns:
        dict: Candidato con mayor apoyo próximo, o motivo de descarte, y franjas probadas.
    """

    referencia = contexto[dentro][ajuste["compatibles"]]
    intentos = []
    ancho = anillo
    while True:
        region = poligono.buffer(ancho)
        if ventana is not None and not ventana.covers(region):
            return {"candidato": None, "motivo": "contexto_recortado", "intentos": intentos}

        exterior = intersects_xy(region, contexto[:, 0], contexto[:, 1]) & ~dentro
        candidatos = buscar_candidatos_exteriores(contexto[exterior], umbral, iteraciones,
                                                   semilla, criterios)
        cercanos = []
        for candidato in candidatos:
            vecinos = contar_vecinos(candidato["puntos"], referencia, criterios["distancia_vecinos"])
            candidato["n_vecinos"] = vecinos
            if vecinos >= 3:
                cercanos.append(candidato)

        intentos.append({"anillo_m": ancho, "n_exterior": int(exterior.sum()),
                         "n_candidatos": len(candidatos), "n_cercanos": len(cercanos)})
        if cercanos:
            elegido = max(cercanos, key=lambda candidato: candidato["n_vecinos"])
            for otro in cercanos:
                angulo = angulo_entre_planos(elegido["ajuste"]["normal"], otro["ajuste"]["normal"])
                distancias = distancias_al_plano(otro["puntos"], elegido["ajuste"]["centro"],
                                                 elegido["ajuste"]["normal"])
                diferente = angulo > 1. or np.percentile(np.abs(distancias), 90) > umbral
                if diferente and elegido["n_vecinos"] < 2 * otro["n_vecinos"]:
                    return {"candidato": None, "motivo": "cubierta_ambigua", "intentos": intentos}
            return {"candidato": elegido, "motivo": None, "intentos": intentos}

        if ancho >= criterios["anillo_maximo"]:
            return {"candidato": None, "motivo": "sin_cubierta_local", "intentos": intentos}
        ancho = min(2 * ancho, criterios["anillo_maximo"])


def decidir_inclinacion(contexto: np.ndarray, poligono, dentro: np.ndarray,
                        umbral: float, umbral_exterior: float, anillo: float,
                        iteraciones: int, semilla: int, criterios: dict, ventana=None) -> dict:
    """Acepta una inclinación compatible con cubierta o conserva el valor supuesto.

    Params:
        contexto: Puntos XYZ completos del recorte.
        poligono: Detección original, sin desplazarla para mejorar el ajuste.
        dentro: Máscara de puntos interiores.
        umbral: Tolerancia del ajuste interior.
        umbral_exterior: Tolerancia del ajuste exterior.
        anillo: Anchura inicial de búsqueda exterior.
        iteraciones: Pruebas por ajuste RANSAC.
        semilla: Semilla base reproducible.
        criterios: Diccionario CRITERIOS con los límites elegidos.
        ventana: Extensión geográfica disponible, si se conoce.

    Returns:
        dict: Ajustes para dibujar y decisión serializable con medidas y motivos.
    """

    decision = {
        "inclinacion_estimada": None,
        "inclinacion_utilizada": criterios["inclinacion_predeterminada"],
        "origen": "predeterminada", "motivo": "sin_plano_interior",
        "calidad_interior": None, "estabilidad_interior": None,
        "cubierta": None, "intentos_exteriores": [],
    }
    resultado = {"ajuste": None, "cubierta": None, "decision": decision}
    puntos = contexto[dentro]
    try:
        ajuste = buscar_plano(puntos, umbral, iteraciones, semilla)
    except ValueError as error:
        decision["detalle"] = str(error)
        return resultado

    resultado["ajuste"] = ajuste
    calidad = evaluar_calidad(puntos, ajuste, criterios, criterios["porcentaje_minimo"])
    decision["calidad_interior"] = calidad
    if calidad["motivo"]:
        decision["motivo"] = calidad["motivo"]
        return resultado

    estabilidad = evaluar_estabilidad(puntos, ajuste, umbral, iteraciones, semilla, criterios)
    decision["estabilidad_interior"] = estabilidad
    if not estabilidad["estable"]:
        decision["motivo"] = "interior_inestable"
        return resultado

    busqueda = buscar_cubierta(contexto, poligono, dentro, ajuste, umbral_exterior,
                              anillo, iteraciones, semilla, criterios, ventana)
    decision["intentos_exteriores"] = busqueda["intentos"]
    if busqueda["motivo"]:
        decision["motivo"] = busqueda["motivo"]
        return resultado

    candidato = busqueda["candidato"]
    cubierta = candidato["ajuste"]
    resultado["cubierta"] = candidato
    estabilidad = evaluar_estabilidad(candidato["puntos"], cubierta, umbral_exterior,
                                       iteraciones, semilla, criterios)
    angulo = angulo_entre_planos(ajuste["normal"], cubierta["normal"])
    distancias = distancias_al_plano(puntos[ajuste["compatibles"]], cubierta["centro"], cubierta["normal"])
    separacion = float(np.percentile(np.abs(distancias), 90))
    decision["cubierta"] = {
        "centro": cubierta["centro"].tolist(), "normal": cubierta["normal"].tolist(),
        "calidad": candidato["calidad"], "estabilidad": estabilidad,
        "n_vecinos": candidato["n_vecinos"], "angulo_entre_planos_grados": angulo,
        "separacion_p90_m": separacion,
    }

    if not estabilidad["estable"]:
        decision["motivo"] = "cubierta_inestable"
    elif angulo > criterios["angulo_maximo"]:
        decision["motivo"] = "orientaciones_distintas"
    elif separacion > criterios["separacion_maxima"]:
        decision["motivo"] = "superficies_separadas"
    else:
        decision["inclinacion_estimada"] = calcular_inclinacion(ajuste["normal"])
        decision["inclinacion_utilizada"] = decision["inclinacion_estimada"]
        decision["origen"] = "estimada"
        decision["motivo"] = "compatible_con_cubierta"

    return resultado


