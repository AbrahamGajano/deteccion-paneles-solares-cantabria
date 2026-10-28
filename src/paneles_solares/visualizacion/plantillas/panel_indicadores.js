/* El HTML contiene datos y agregados. Cambiar filtros no modifica los originales. */
const datos = JSON.parse(document.getElementById('datos').textContent);
const elemento = id => document.getElementById(id);
const numero = (valor, decimales = 0) => valor == null ? 'Sin dato' :
    Number(valor).toLocaleString('es-ES', { maximumFractionDigits: decimales });
const texto = valor => String(valor ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const etiquetas = {
    comprobar_edificio: 'Edificio asignado', resolver_atribucion: 'Atribución dudosa',
    comprobar_emplazamiento_exterior: 'Sin coincidencia cartográfica',
    conflicto_entre_edificios: 'Conflicto entre edificios', presente: 'Presencia confirmada',
    ausente: 'Ausencia confirmada', desconocido: 'Desconocido',
    detectado_sin_confirmar: 'Detección sin confirmar', vinculo_pendiente: 'Vínculo pendiente',
    pendiente_revision: 'Dibujo pendiente de revisión', sin_presencia: 'Presencia no confirmada',
    agrupacion_confirmada: 'Agrupación confirmada',
    acotado: 'Intervalo acotado', sin_ausencia_previa: 'Sin ausencia previa',
    presencia_por_confirmar: 'Presencia por confirmar',
    revisar_cambio_o_contradiccion: 'Revisar cambio o contradicción',
};
const nombres = Object.fromEntries(datos.municipios.features.map(f => [f.properties.codigo, f.properties.municipio]));
nombres.fuera = 'Sin adscripción municipal';
const observaciones = new Map();
for (const fila of datos.serie) {
    if (!observaciones.has(fila.id_instalacion)) observaciones.set(fila.id_instalacion, {});
    observaciones.get(fila.id_instalacion)[fila.anio] = fila;
}
const clave = (...valores) => JSON.stringify(valores);
const resumenes = new Map(datos.resumen.map(r => [clave(r.modo, r.anio, r.uso, r.revision, r.codigo), r]));
const candidatosPorId = new Map(datos.candidatos.map(c => [c.id_instalacion, c]));
let modo = 'inventario';
let seleccion = null;
let mapa, municipios, puntos, mascaras, contexto, ortofoto;
let limiteEscala = 1;
let pagina = 0;
const filasPorPagina = 20;

function opciones(id, valores, inicial) {
    elemento(id).innerHTML = valores.map(([valor, titulo]) =>
        `<option value="${texto(valor)}">${texto(titulo)}</option>`).join('');
    elemento(id).value = inicial;
}

function filtros() {
    return Object.fromEntries(['anio', 'municipio', 'uso', 'revision', 'metrica'].map(id => [id, elemento(id).value]));
}

function resumen(anio, codigo, vista = modo) {
    const f = filtros();
    return resumenes.get(clave(vista, Number(anio), f.uso, f.revision, codigo));
}

function candidatosFiltrados(vista = modo) {
    const f = filtros();
    return datos.candidatos.filter(c => (vista !== 'muestra' || c.muestra) &&
        (f.municipio === '*' || c.codigo === f.municipio) &&
        (f.uso === '*' || c.uso === f.uso) && (f.revision === '*' || c.revision === f.revision));
}

function fuenteCampania(anio) {
    const fuentes = datos.fuentes[anio] || [];
    const modelo = fuentes.some(fuente => fuente.startsWith('modelo'));
    if (modelo && fuentes.includes('labelme')) return 'Superficies manuales y estimadas por el modelo.';
    if (modelo) return 'Superficie estimada por el modelo.';
    if (fuentes.includes('labelme')) return 'Superficie anotada manualmente.';
    return 'Sin superficie delimitada en esta campaña.';
}

function barras(id, filas, unidad = '') {
    const maximo = Math.max(1, ...filas.map(f => f[1] ?? 0));
    elemento(id).innerHTML = `<div class="barras">${filas.map(([nombre, valor]) =>
        `<div class="barra"><span>${texto(nombre)}</span><div><i style="width:${100 * (valor ?? 0) / maximo}%"></i></div>` +
        `<strong>${numero(valor)}${valor == null ? '' : unidad}</strong></div>`).join('')}</div>`;
}

function cambiarVista(vista) {
    for (const nombre of ['resumen', 'analisis']) {
        const activa = nombre === vista;
        elemento(`vista-${nombre}`).hidden = !activa;
        elemento(`tab-${nombre}`).classList.toggle('activo', activa);
        elemento(`tab-${nombre}`).setAttribute('aria-selected', String(activa));
    }
    if (vista === 'resumen' && mapa) requestAnimationFrame(() => mapa.invalidateSize());
}

function filasSeleccionadas() {
    const anio = Number(elemento('anio').value);
    return candidatosFiltrados().map(c => ({ ...c, ...observaciones.get(c.id_instalacion)[anio] }));
}

function percentil(ordenadas, p) {
    if (!ordenadas.length) return null;
    const posicion = (ordenadas.length - 1) * p;
    const abajo = Math.floor(posicion), arriba = Math.ceil(posicion);
    return ordenadas[abajo] + (ordenadas[arriba] - ordenadas[abajo]) * (posicion - abajo);
}

function nombreUso(uso) {
    const nombres = {
        'Residencial en vivienda no colectiva': 'Residencial no colectivo',
        'Residencial en vivienda colectiva': 'Residencial colectivo',
        'Industrial o manufacturero o logístico-almacenaje': 'Industrial / logística',
        'Instalación agrícola o ganadera': 'Agrícola / ganadero',
        'Turístico u hostelero u ocio y espectáculos': 'Turístico / hostelero / ocio',
        'Servicios administrativos o de oficinas': 'Oficinas / administración',
        'Sanitario y asistencial': 'Sanitario / asistencial',
        'Sin dato cartográfico': 'Sin dato',
    };
    return nombres[uso] || uso;
}

function nombreFuente(fuente) {
    if (fuente === 'labelme') return 'Manual';
    if (fuente === 'seleccion_anterior') return 'Revisión anterior';
    if (fuente?.startsWith('modelo')) return 'Modelo';
    return '';
}

function dibujarUsos(filas) {
    if (!filas.length) {
        elemento('usos').innerHTML = '<p class="nota">Sin superficies positivas en esta selección.</p>';
        return;
    }
    const grupos = new Map();
    for (const fila of filas) {
        const uso = fila.uso.includes(';') ? 'Varios usos' : nombreUso(fila.uso);
        grupos.set(uso, (grupos.get(uso) || 0) + 1);
    }
    const conocidos = [...grupos].filter(([uso]) => uso !== 'Sin dato').sort((a, b) => b[1] - a[1]);
    const partes = conocidos.slice(0, 5);
    const otros = conocidos.slice(5).reduce((suma, fila) => suma + fila[1], 0);
    if (otros) partes.push(['Otros usos', otros]);
    if (grupos.has('Sin dato')) partes.push(['Sin dato', grupos.get('Sin dato')]);
    const colores = ['#117c78', '#3577a5', '#ba7b31', '#805ca3', '#7f984c', '#b46e7b'];
    let inicio = 0;
    const segmentos = [], leyenda = [];
    partes.forEach(([uso, n], i) => {
        const color = uso === 'Sin dato' ? '#b6c2c0' : colores[i];
        const porcentaje = 100 * n / filas.length;
        segmentos.push(`${color} ${inicio}% ${inicio + porcentaje}%`);
        inicio += porcentaje;
        leyenda.push(`<li><i style="background:${color}"></i><span>${texto(uso)}</span>` +
            `<strong>${numero(n)} · ${numero(porcentaje, 1)} %</strong></li>`);
    });
    elemento('usos').innerHTML = `<div class="reparto"><div class="anillo" aria-hidden="true" ` +
        `style="background:conic-gradient(${segmentos.join(',')})"></div><ul>${leyenda.join('')}</ul></div>`;
}

function actualizarInstalaciones() {
    const filas = filasSeleccionadas();
    const positivas = filas.filter(f => f.superficie_horizontal_m2 > 0);
    const areas = positivas.map(f => f.superficie_horizontal_m2).sort((a, b) => a - b);
    const conUso = positivas.filter(f => f.uso !== 'Sin dato cartográfico').length;
    const tarjetas = [
        [numero(positivas.length), 'agrupaciones con superficie', 'En la campaña seleccionada'],
        [numero(percentil(areas, .5), 1), 'm² · superficie mediana', 'La mitad tiene una superficie menor o igual'],
        [numero(percentil(areas, .9), 1), 'm² · percentil 90', 'El 90 % tiene una superficie menor o igual'],
        [positivas.length ? `${numero(100 * conUso / positivas.length, 1)} %` : 'Sin dato', 'con uso cartográfico', 'Referencia CBT23, no clasificación manual'],
    ];
    elemento('resumen-instalaciones').innerHTML = tarjetas.map(([valor, titulo, nota]) =>
        `<div class="total"><strong>${valor}</strong><span>${titulo}</span><small>${nota}</small></div>`).join('');
    const limites = [0, 5, 10, 20, 50, 100, 250, 1000, Infinity];
    const rangos = ['< 5 m²', '5 a < 10', '10 a < 20', '20 a < 50', '50 a < 100', '100 a < 250', '250 a < 1000', '≥ 1000 m²'];
    barras('tamanos', rangos.map((rango, i) => [rango,
        areas.filter(a => a >= limites[i] && a < limites[i + 1]).length]));
    const sinArea = filas.filter(f => f.estado === 'presente' && f.superficie_horizontal_m2 == null).length;
    elemento('tamanos-nota').textContent = `${numero(positivas.length)} agrupaciones con área positiva. ` +
        `${numero(sinArea)} presencias sin área delimitada quedan fuera. ${fuenteCampania(filtros().anio)}`;
    dibujarUsos(positivas);
    elemento('usos-nota').textContent = 'Reparto por número de agrupaciones con superficie, no por m². ' +
        'Uso de la edificación según CBT23/LULC23, aplicado a todas las campañas. No identifica chalets, urbanizaciones ni titularidad pública.';
    const soportes = [
        ['Un edificio claro', 'un_edificio'],
        ['Varios edificios claros', 'varios_edificios'],
        ['Parcial o dudosa', 'atribucion_dudosa'],
        ['Sin coincidencia', 'sin_coincidencia'],
    ].map(([nombre, soporte]) => [nombre, positivas.filter(f => f.soporte === soporte).length]);
    barras('soportes', soportes.filter(([, n]) => n > 0));
    if (!positivas.length) elemento('soportes').innerHTML = '<p class="nota">Sin superficies positivas en esta selección.</p>';
    actualizarTabla();
}

function filasTabla() {
    const busqueda = elemento('buscar-instalacion').value.trim().toLocaleLowerCase('es');
    return filasSeleccionadas().filter(f =>
        `${f.id_instalacion} ${nombres[f.codigo]}`.toLocaleLowerCase('es').includes(busqueda)
    ).sort((a, b) => (b.superficie_horizontal_m2 ?? -1) - (a.superficie_horizontal_m2 ?? -1));
}

function actualizarTabla() {
    const filas = filasTabla();
    pagina = Math.min(pagina, Math.max(0, Math.ceil(filas.length / filasPorPagina) - 1));
    const inicio = pagina * filasPorPagina;
    const visibles = filas.slice(inicio, inicio + filasPorPagina);
    elemento('tabla-instalaciones').innerHTML = visibles.length ? visibles.map(f =>
        `<tr><td><button class="identificador" type="button" data-instalacion="${texto(f.id_instalacion)}">${texto(f.id_instalacion)}</button>` +
        `<small>${texto(nombres[f.codigo])}</small></td><td title="${texto(f.uso)}">${texto(nombreUso(f.uso))}</td>` +
        `<td>${texto(etiquetas[f.estado])}<small>${texto(nombreFuente(f.fuente_superficie))}</small></td>` +
        `<td>${numero(f.superficie_horizontal_m2, 1)}</td></tr>`).join('') :
        '<tr><td colspan="4">Sin resultados para esta selección.</td></tr>';
    elemento('filas-instalaciones').textContent = filas.length ?
        `${numero(inicio + 1)}–${numero(inicio + visibles.length)} de ${numero(filas.length)} · campaña ${filtros().anio}` : '0 resultados';
    elemento('pagina-anterior').disabled = pagina === 0;
    elemento('pagina-siguiente').disabled = inicio + filasPorPagina >= filas.length;
}

function descargarCSV(nombre, columnas, filas) {
    const csv = [columnas, ...filas].map(f => f.map(v => `"${String(v ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8' }));
    const enlace = document.createElement('a');
    enlace.href = url;
    enlace.download = nombre;
    document.body.appendChild(enlace);
    enlace.click();
    enlace.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function descargarResumen(energia = false) {
    const columnas = energia ? [
        'modo', 'anio', 'uso', 'revision', 'codigo', 'superficie_m2',
        'produccion_min_mwh', 'produccion_central_mwh', 'produccion_max_mwh',
        'potencia_central_kwp', 'superficie_antiguedad_supuesta_m2',
    ] : Object.keys(datos.resumen[0] || {});
    descargarCSV(energia ? 'produccion_estimada.csv' : 'resumen_municipal.csv',
        columnas, datos.resumen.map(f => columnas.map(c => f[c])));
}

function descargarInstalaciones() {
    const columnas = ['id_instalacion', 'anio', 'municipio', 'uso_cartografico_2023', 'edificios', 'parcelas',
        'estado', 'superficie_horizontal_m2', 'fuente_superficie', 'instalacion_desde', 'instalacion_hasta',
        'estado_periodo', 'instalacion_confirmada', 'revision'];
    const filas = filasTabla().map(f => [f.id_instalacion, f.anio, nombres[f.codigo], f.uso, f.edificios,
        f.parcelas, f.estado, f.superficie_horizontal_m2, f.fuente_superficie,
        f.instalacion_desde, f.instalacion_hasta, f.estado_periodo, f.instalacion_confirmada, f.revision]);
    descargarCSV(`instalaciones_${filtros().anio}.csv`, columnas, filas);
}

function actualizarGraficos() {
    const f = filtros();
    const cohorte = candidatosFiltrados('muestra');
    const campo = f.metrica === 'con_mascara' ? 'presentes' : f.metrica;
    const titulos = { superficie_m2: 'Superficie de la muestra comparable', presentes: 'Presencia en la muestra comparable', produccion_central_mwh: 'Producción estimada · MWh/año' };
    elemento('evolucion-titulo').textContent = titulos[campo];
    barras('evolucion', datos.comparables.map(anio => [anio, resumen(anio, f.municipio, 'muestra')?.[campo]]), campo === 'superficie_m2' ? ' m²' : '');
    elemento('cohorte').textContent = datos.comparables.length ?
        `${numero(cohorte.length)} agrupaciones con área comparable en ${datos.comparables.join(', ')}.` :
        'Se necesitan al menos dos campañas con superficies revisadas para comparar.';

    const elegidos = candidatosFiltrados();
    const intervalos = new Map();
    for (const c of elegidos.filter(c => c.estado_periodo === 'acotado')) {
        const nombre = `${c.instalacion_desde}–${c.instalacion_hasta}`;
        intervalos.set(nombre, (intervalos.get(nombre) ?? 0) + 1);
    }
    const filas = [...intervalos].sort((a, b) => a[0].localeCompare(b[0]));
    barras('periodos', filas);
    const acotados = filas.reduce((s, f) => s + f[1], 0);
    elemento('periodos-nota').textContent = `${numero(acotados)} intervalos acotados; ${numero(elegidos.filter(c => c.estado_periodo === 'sin_ausencia_previa').length)} sin ausencia previa. ${numero(elegidos.filter(c => !['acotado', 'sin_ausencia_previa'].includes(c.estado_periodo)).length)} por confirmar o resolver. El intervalo va desde la última ausencia hasta la primera presencia.`;

    if (!elemento('intervalo').value) {
        barras('cambios', []);
        elemento('cambios-nota').textContent = 'Faltan campañas con áreas comparables.';
        return;
    }
    const [antes, despues] = elemento('intervalo').value.split(',');
    const parejas = cohorte.map(c => {
        const filas = observaciones.get(c.id_instalacion);
        return [filas[antes].superficie_horizontal_anotada_m2, filas[despues].superficie_horizontal_anotada_m2];
    });
    const conPaneles = parejas.filter(([a, b]) => a > 0 || b > 0);
    const diferencias = conPaneles.map(([a, b]) => b - a);
    const limites = [-Infinity, -20, -5, 5, 20, Infinity];
    const rangos = ['< −20 m²', '−20 a −5', '−5 a +5', '+5 a +20', '≥ +20 m²'];
    barras('cambios', rangos.map((nombre, i) => [nombre,
        diferencias.filter(d => d >= limites[i] && d < limites[i + 1]).length]));
    const aparecen = conPaneles.filter(([a, b]) => a === 0 && b > 0).length;
    const desaparecen = conPaneles.filter(([a, b]) => a > 0 && b === 0).length;
    elemento('cambios-nota').textContent = `${numero(conPaneles.length)} agrupaciones con paneles en alguna de las dos fechas. ` +
        `${numero(parejas.length - conPaneles.length)} ausentes en ambas fechas, separados del histograma. ` +
        `Aparecen: ${numero(aparecen)}; desaparecen: ${numero(desaparecen)}. ` +
        'Área horizontal anotada: un cambio pequeño puede proceder del dibujo o de la imagen; no prueba una ampliación.';
}

function actualizarTotales() {
    const f = filtros();
    const r = resumen(f.anio, f.municipio);
    const tarjetas = [
        [numero(r?.superficie_m2, 1), 'm² horizontales observados', fuenteCampania(f.anio)],
        [numero(r?.con_mascara ?? 0), 'agrupaciones con paneles', 'Con superficie delimitada en esta campaña'],
        [numero(r?.ausentes ?? 0), 'ausencias confirmadas', 'En la campaña seleccionada'],
        [numero(r?.sin_estado_resuelto ?? 0), 'estados pendientes', 'Incluye detecciones sin confirmar y vínculos pendientes'],
    ];
    elemento('totales').innerHTML = tarjetas.map(([valor, titulo, nota]) =>
        `<div class="total"><strong>${valor}</strong><span>${titulo}</span><small>${nota}</small></div>`).join('');
    elemento('alcance').textContent = modo === 'muestra' ?
        `Muestra comparable de ${datos.comparables.join(', ')}: mismas agrupaciones con áreas revisadas en todas las fechas.` :
        `Campaña ${f.anio}. ${fuenteCampania(f.anio)} Inventario seleccionado; la cobertura cambia entre campañas.`;
}

function actualizarEnergia() {
    if (!datos.energia) return;
    const f = filtros();
    const r = resumen(f.anio, f.municipio);
    const tarjetas = [
        [numero(r?.produccion_central_mwh), 'MWh/año · escenario central', 'Edad aproximada y eficiencia según época'],
        [numero(r?.produccion_min_mwh), 'MWh/año · conservador', 'Orientación oeste, pérdidas y envejecimiento mayores'],
        [numero(r?.produccion_max_mwh), 'MWh/año · favorable', 'Permite módulos nuevos en toda el área observada'],
        [numero(r?.potencia_central_kwp != null ? r.potencia_central_kwp / 1000 : null, 2), 'MWp · potencia nominal estimada', 'Potencia inicial; la producción sí descuenta degradación'],
    ];
    elemento('energia-totales').innerHTML = tarjetas.map(([valor, titulo, nota]) =>
        `<div class="total"><strong>${valor}</strong><span>${titulo}</span><small>${nota}</small></div>`).join('');
    const unidad = f.municipio === '*' ? 'inventario seleccionado' : nombres[f.municipio];
    elemento('energia-alcance').textContent = `Campaña ${f.anio} · ${unidad}. ` +
        'Producción anual representativa sobre la superficie disponible, con clima medio de PVGIS. ' +
        fuenteCampania(f.anio);
    const rendimiento = r?.superficie_m2 > 0 ? r.produccion_central_mwh * 1000 / r.superficie_m2 : null;
    const sinFecha = r?.superficie_m2 > 0 ? 100 * r.superficie_antiguedad_supuesta_m2 / r.superficie_m2 : null;
    if (rendimiento == null) {
        elemento('energia-edades').textContent = 'No hay superficie positiva delimitada en esta selección.';
        return;
    }
    elemento('energia-edades').textContent = `Central: ${numero(rendimiento)} kWh por m² horizontal y año. ` +
        `${numero(sinFecha, 1)} % del área necesita el supuesto de antigüedad de hasta 25 años antes de la primera imagen. ` +
        'Los escenarios no son intervalos de confianza y no fechan ampliaciones o sustituciones.';
}

function mostrarSupuestosEnergia() {
    const h = datos.energia;
    document.querySelector('.bloque-energia').hidden = !h;
    if (!h) return;
    const nombres = { min: 'Conservador', central: 'Central', max: 'Favorable' };
    const filas = Object.entries(h.escenarios).map(([nombre, e]) => `<tr><td>${nombres[nombre]}</td>` +
        `<td>${e.inclinacion_grados}°</td><td>${e.orientacion_grados === 90 ? 'Oeste' : 'Sur'}</td>` +
        `<td>${e.perdidas_operativas_pct} %</td><td>${numero(100 * e.degradacion_anual, 1)} %/año</td>` +
        `<td>× ${numero(e.factor_eficiencia, 2)}</td></tr>`).join('');
    const etiquetas = { radiacion: 'Radiación', superficie: 'Superficie', eficiencia: 'Tecnología y época',
        antiguedad: 'Edad central', ampliaciones: 'Ampliaciones y renovaciones', perdidas: 'Pérdidas',
        periodo: 'Año representativo', limites: 'Alcance del rango' };
    elemento('energia-supuestos').innerHTML = '<div class="tabla-supuestos"><table><thead><tr>' +
        '<th>Escenario</th><th>Inclinación</th><th>Orientación</th><th>Pérdidas</th><th>Degradación</th><th>Eficiencia de época</th>' +
        `</tr></thead><tbody>${filas}</tbody></table></div>` +
        Object.entries(etiquetas).map(([clave, titulo]) => `<p><strong>${titulo}.</strong> ${texto(h[clave])}</p>`).join('') +
        '<p><strong>Eficiencias de referencia.</strong> ' + Object.entries(h.eficiencia_por_anio).map(([a, e]) => `${a}: ${numero(e * 100, 1)} %`).join(' · ') + '</p>' +
        '<p><strong>Fuentes:</strong> ' + Object.entries(h.fuentes).map(([nombre, url]) => `<a href="${texto(url)}" target="_blank" rel="noopener">${texto(nombre)}</a>`).join(' · ') + '</p>';
}

function maximoEscala() {
    const anios = modo === 'muestra' ? datos.comparables : datos.anios;
    return Math.max(1, ...datos.resumen.filter(r => r.modo === modo && anios.includes(r.anio) &&
        r.uso === '*' && r.revision === '*' && !['*', 'fuera'].includes(r.codigo))
        .map(r => r[filtros().metrica] ?? 0));
}

function color(valor) {
    if (valor == null) return '#d3dcda';
    if (valor === 0) return '#ffffff';
    const t = Math.sqrt(Math.min(1, valor / limiteEscala));
    const a = [224, 240, 235], b = [6, 67, 66];
    return `rgb(${a.map((v, i) => Math.round(v + t * (b[i] - v))).join(',')})`;
}

function estiloMunicipio(feature) {
    const f = filtros();
    const r = resumen(f.anio, feature.properties.codigo);
    const valor = f.metrica === 'con_mascara' && r?.superficie_m2 == null ? null : r?.[f.metrica];
    return { color: f.municipio === feature.properties.codigo ? '#142d39' : '#7d9691',
        weight: f.municipio === feature.properties.codigo ? 2.8 : .7,
        fillColor: color(valor), fillOpacity: mapa?.getZoom() >= 15 ? .05 : .74 };
}

function verFicha() {
    const c = candidatosPorId.get(seleccion);
    if (!c) {
        elemento('detalle').innerHTML = '<p>Selecciona un punto para ver el detalle por campaña.</p>';
        return;
    }
    const filas = observaciones.get(c.id_instalacion);
    const periodo = c.estado_periodo === 'acotado' ? `${c.instalacion_desde} < aparición ≤ ${c.instalacion_hasta}` :
        c.estado_periodo === 'sin_ausencia_previa' ? `Presencia en ${c.instalacion_hasta}; inicio desconocido` : etiquetas[c.estado_periodo];
    const idsEdificio = (c.edificios || '').split(';').filter(Boolean);
    const idsParcela = (c.parcelas || '').split(';').filter(Boolean);
    const tipo = c.instalacion_confirmada ? 'Instalación confirmada' : 'Agrupación propuesta';
    const referencias = datos.contexto.parcelas_2023.features.filter(f => idsParcela.includes(f.properties.id_parcela))
        .map(f => f.properties.referencia_catastral).filter(Boolean);
    elemento('detalle').innerHTML = `<p class="identificador">${texto(c.id_instalacion)}</p>` +
        `<span class="etiqueta">${tipo} · ${texto(etiquetas[c.revision] || c.revision)}</span>` +
        `<p><strong>${texto(nombres[c.codigo])}</strong><br>${texto(c.uso)}</p>` +
        `<p>${texto(periodo)}${c.ausencia_posterior ? '<br>Hay ausencia posterior: revisar continuidad.' : ''}</p>` +
        '<table><thead><tr><th>Año / estado</th><th>m² horizontales</th></tr></thead><tbody>' +
        datos.anios.map(anio => {
            const o = filas[anio];
            const manual = o.superficie_horizontal_anotada_m2;
            const area = manual ?? o.superficie_horizontal_modelo_m2;
            const fuente = manual != null ? (manual === 0 ? 'ausencia' : 'manual') : area != null ? 'modelo' : '';
            return `<tr><td><strong>${anio}</strong><br>${texto(etiquetas[o.estado])}` +
                `${o.conflicto ? '<br>⚠ Conflicto' : ''}</td><td>${numero(area, 1)}<br>${fuente}` +
                `${manual != null && !o.superficie_comparable ? '<br>no comparable' : ''}</td></tr>`;
        }).join('') + '</tbody></table>' +
        `<p>${idsEdificio.length ? `${idsEdificio.length} unidad(es) edificada(s) de referencia` : 'Sin edificio atribuible; puede ser exterior o estar sin resolver'} · ${idsParcela.length} parcela(s)</p>` +
        `<details><summary>Identificadores del contexto</summary><p class="identificador">${texto(idsEdificio.join(' · '))}<br>${texto(idsParcela.join(' · '))}<br>RC: ${texto(referencias.join(' · '))}</p></details>`;
}

function dibujarDetalle() {
    if (!mapa) return;
    const f = filtros();
    const elegidos = candidatosFiltrados();
    const ids = new Set(elegidos.map(c => c.id_instalacion));
    puntos.clearLayers();
    if (mapa.getZoom() >= 12) {
        for (const c of elegidos) {
            const o = observaciones.get(c.id_instalacion)[f.anio];
            const positivo = o.superficie_horizontal_anotada_m2 > 0 || o.superficie_horizontal_modelo_m2 > 0;
            L.circleMarker([c.lat, c.lon], { radius: c.id_instalacion === seleccion ? 7 : 4,
                color: positivo ? (o.superficie_horizontal_modelo_m2 > 0 ? '#b16e00' : '#1666aa') : '#748587',
                weight: 1.5, fillOpacity: positivo ? .85 : .1 })
                .bindTooltip(`${texto(c.id_instalacion)} · ${texto(etiquetas[o.estado])}`)
                .on('click', () => elegirCandidato(c.id_instalacion)).addTo(puntos);
        }
    }
    mascaras.clearLayers();
    const cerca = mapa.getBounds().pad(.1);
    for (const feature of datos.paneles.features) {
        const p = feature.properties;
        if (p.anio !== Number(f.anio) || !ids.has(p.id_instalacion)) continue;
        if (p.id_instalacion !== seleccion && mapa.getZoom() < 16) continue;
        const c = candidatosPorId.get(p.id_instalacion);
        if (p.id_instalacion !== seleccion && !cerca.contains([c.lat, c.lon])) continue;
        L.geoJSON(feature, { style: { color: p.fuente_geometria === 'labelme' ? '#1974c9' : '#f8b928',
            weight: p.id_instalacion === seleccion ? 3 : 1.5, fillOpacity: .3 } })
            .bindTooltip(`${texto(p.atribucion)} · ${texto(p.fuente_geometria)}`)
            .on('click', () => elegirCandidato(p.id_instalacion)).addTo(mascaras);
    }
    contexto.clearLayers();
    const c = candidatosPorId.get(seleccion);
    if (c) {
        for (const [capa, columna, ids] of [
            ['edificios_2023', 'id_edificio', (c.edificios || '').split(';')],
            ['parcelas_2023', 'id_parcela', (c.parcelas || '').split(';')],
        ]) {
            L.geoJSON(datos.contexto[capa], {
                filter: feature => ids.includes(feature.properties[columna]),
                style: { color: '#ae65bc', weight: 2, fillOpacity: 0, dashArray: capa === 'parcelas_2023' ? '6 5' : null },
                onEachFeature: (feature, capa) => capa.bindTooltip(texto(feature.properties.referencia_catastral || feature.properties[columna])),
            }).addTo(contexto);
        }
    }
}

function elegirCandidato(id) {
    seleccion = id;
    verFicha();
    const c = candidatosPorId.get(id);
    if (mapa) mapa.setView([c.lat, c.lon], Math.max(19, mapa.getZoom()));
    dibujarDetalle();
}

function actualizar() {
    if (seleccion && !candidatosFiltrados().some(c => c.id_instalacion === seleccion)) seleccion = null;
    actualizarTotales();
    actualizarEnergia();
    actualizarGraficos();
    pagina = 0;
    actualizarInstalaciones();
    verFicha();
    const f = filtros();
    const maximo = maximoEscala();
    limiteEscala = maximo;
    elemento('leyenda-titulo').textContent = {
        superficie_m2: 'Superficie horizontal observada · m²', con_mascara: 'Agrupaciones con paneles',
        produccion_central_mwh: 'Producción estimada del inventario · MWh/año',
    }[f.metrica];
    elemento('escala').innerHTML = [0, .25, 1].map(t => `<span>${numero(maximo * t)}</span>`).join('');
    if (!mapa) return;
    municipios.setStyle(estiloMunicipio);
    ortofoto.setUrl(`https://geoservicios.cantabria.es/inspire/rest/services/Ortofoto_${f.anio}_WebMercator/MapServer/tile/{z}/{y}/{x}`);
    dibujarDetalle();
}

function crearMapa() {
    mapa = L.map('mapa', { preferCanvas: true, maxZoom: 22 }).setView([43.20, -4.04], 9);
    const sinFondo = L.layerGroup().addTo(mapa);
    const calles = L.tileLayer('https://tms-ign-base.idee.es/1.0.0/IGNBaseTodo/{z}/{x}/{-y}.jpeg', {
        maxNativeZoom: 19, maxZoom: 22,
        attribution: '<a href="https://www.ign.es">© Instituto Geográfico Nacional</a>' });
    ortofoto = L.tileLayer('', { maxNativeZoom: 19, maxZoom: 22,
        attribution: 'Ortofoto de la campaña · Gobierno de Cantabria / IGN-CNIG' });
    municipios = L.geoJSON(datos.municipios, {
        style: estiloMunicipio,
        onEachFeature: (feature, capa) => {
            capa.bindTooltip(() => {
                const r = resumen(filtros().anio, feature.properties.codigo);
                return `<strong>${texto(feature.properties.municipio)}</strong><br>` +
                    `${numero(r?.superficie_m2, 1)} m² observados<br>${numero(r?.con_mascara ?? 0)} agrupaciones con paneles`;
            });
            capa.on('click', () => {
                elemento('municipio').value = feature.properties.codigo;
                elemento('municipio').dispatchEvent(new Event('change'));
            });
        },
    }).addTo(mapa);
    puntos = L.layerGroup().addTo(mapa);
    contexto = L.layerGroup().addTo(mapa);
    mascaras = L.layerGroup().addTo(mapa);
    L.control.layers({ 'Sin fondo · sin conexión': sinFondo,
        'Mapa de calles · Internet': calles, 'Ortofoto de la campaña · Internet': ortofoto },
        { 'Máscaras': mascaras, 'Edificio y parcela · 2023': contexto }, { collapsed: false }).addTo(mapa);
    mapa.on('moveend', () => { municipios.setStyle(estiloMunicipio); dibujarDetalle(); });
}

opciones('anio', datos.anios.map(a => [a, a]), String(datos.anios.at(-1)));
const intervalos = [];
for (let i = 0; i < datos.comparables.length; i++) {
    for (let j = i + 1; j < datos.comparables.length; j++) {
        const antes = datos.comparables[i], despues = datos.comparables[j];
        intervalos.push([`${antes},${despues}`, `${antes} → ${despues}`]);
    }
}
opciones('intervalo', intervalos, intervalos[0]?.[0] || '');
elemento('intervalo').disabled = !intervalos.length;
elemento('muestra').disabled = !datos.comparables.length;
opciones('municipio', [['*', 'Todo el inventario'], ...Object.entries(nombres).sort((a, b) => a[1].localeCompare(b[1], 'es'))], '*');
opciones('uso', [['*', 'Todos los usos'], ...[...new Set(datos.candidatos.map(c => c.uso))].sort().map(u => [u, u])], '*');
opciones('revision', [['*', 'Todas las atribuciones'], ...[...new Set(datos.candidatos.map(c => c.revision))].sort().map(r => [r, etiquetas[r] || r])], '*');
for (const id of ['inventario', 'muestra']) {
    elemento(id).addEventListener('click', () => {
        modo = id;
        for (const boton of ['inventario', 'muestra']) elemento(boton).classList.toggle('activo', boton === modo);
        const anio = Number(elemento('anio').value);
        const anios = modo === 'muestra' ? datos.comparables : datos.anios;
        opciones('anio', anios.map(a => [a, a]), String(anios.includes(anio) ? anio : anios.at(-1)));
        actualizar();
    });
}
for (const id of ['anio', 'uso', 'revision', 'metrica', 'intervalo']) elemento(id).addEventListener('change', actualizar);
elemento('municipio').addEventListener('change', () => {
    actualizar();
    if (!mapa) return;
    const codigo = elemento('municipio').value;
    const capa = municipios.getLayers().find(c => c.feature.properties.codigo === codigo);
    if (capa) mapa.fitBounds(capa.getBounds(), { padding: [15, 15] });
    else if (codigo === '*') mapa.fitBounds(municipios.getBounds());
    else {
        const ubicaciones = candidatosFiltrados().map(c => [c.lat, c.lon]);
        if (ubicaciones.length) mapa.fitBounds(ubicaciones, { maxZoom: 17, padding: [25, 25] });
    }
});
elemento('cerrar').addEventListener('click', () => { seleccion = null; verFicha(); dibujarDetalle(); });
for (const vista of ['resumen', 'analisis']) {
    elemento(`tab-${vista}`).addEventListener('click', () => cambiarVista(vista));
}
elemento('buscar-instalacion').addEventListener('input', () => { pagina = 0; actualizarTabla(); });
elemento('pagina-anterior').addEventListener('click', () => { pagina--; actualizarTabla(); });
elemento('pagina-siguiente').addEventListener('click', () => { pagina++; actualizarTabla(); });
elemento('descargar-instalaciones').addEventListener('click', descargarInstalaciones);
elemento('descargar-resumen').addEventListener('click', () => descargarResumen());
elemento('descargar-energia').addEventListener('click', () => descargarResumen(true));
elemento('tabla-instalaciones').addEventListener('click', evento => {
    const boton = evento.target.closest('[data-instalacion]');
    if (!boton) return;
    cambiarVista('resumen');
    elegirCandidato(boton.dataset.instalacion);
});
crearMapa();
mostrarSupuestosEnergia();
actualizar();
