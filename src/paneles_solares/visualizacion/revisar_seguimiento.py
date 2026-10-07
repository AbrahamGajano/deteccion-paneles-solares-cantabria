"""Visor local de campañas independientes, con capas y navegación sincronizadas."""

import json
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from tkinter import messagebox, simpledialog, ttk

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from shapely import from_wkb, union_all
from shapely.geometry import Point, box

from paneles_solares.datos.seguimiento import (
    DECISIONES,
    ETIQUETAS,
    RELACIONES,
    SOPORTES,
    AlmacenSeguimiento,
)
from paneles_solares.modelos.resumir_seguimiento import (
    SALIDA,
    generar_resumen,
    trazar_geometria,
)
from paneles_solares.rutas import ruta_proyecto

NOMBRES = {
    "presente": "Presencia clara",
    "ausente": "Ausencia clara",
    "dudosa": "Dudoso",
    "no_evaluable": "No evaluable",
    "referencia_incorrecta": "Referencia 2023 incorrecta",
}
BOTONES = {
    "presente": "Hay paneles",
    "ausente": "No hay paneles",
    "dudosa": "No estoy seguro",
    "no_evaluable": "No se puede evaluar",
    "referencia_incorrecta": "No hay en ambas",
}
CAMBIOS = {
    "ampliacion": "Más paneles en 2023",
    "reduccion": "Menos paneles en 2023",
    "distribucion_diferente": "Distribución diferente en 2023",
}


def objetivos_visuales(almacen, sitio, ventanas):
    """Localiza el caso con sus componentes originales y distingue ubicaciones provisionales.

    Args:
        almacen: Base de revisión, consultada sin cambiar geometrías ni observaciones.
        sitio: Emplazamiento seleccionado.
        ventanas: Recortes visibles, incluidos los de contexto.

    Returns:
        Objetivo seleccionado y candidatos vecinos con su geometría de localización.
    """
    marcas = ",".join("?" for _ in ventanas)
    sitios = almacen.filas(
        f"""SELECT DISTINCT e.id,e.geometria,e.origen FROM emplazamientos e
        LEFT JOIN miembros m ON m.emplazamiento=e.id
        LEFT JOIN componentes c ON c.id=m.componente
        WHERE e.id=? OR (e.activo=1 AND (e.ventana IN ({marcas}) OR c.ventana IN ({marcas})))
        ORDER BY e.id""",
        (sitio, *ventanas, *ventanas),
    )
    miembros = almacen.filas(
        f"""SELECT m.emplazamiento,c.geometria FROM miembros m
        JOIN componentes c ON c.id=m.componente JOIN emplazamientos e ON e.id=m.emplazamiento
        WHERE c.anio=2023 AND (e.id=? OR e.ventana IN ({marcas}) OR c.ventana IN ({marcas}))""",
        (sitio, *ventanas, *ventanas),
    )
    grupos = {}
    for componente in miembros:
        grupos.setdefault(componente["emplazamiento"], []).append(from_wkb(componente["geometria"]))
    objetivos = []
    for fila in sitios:
        componentes = grupos.get(fila["id"], [])
        provisional = not componentes and fila["origen"] in ("piloto", "inventario_2023")
        if fila["id"] != sitio and provisional:
            continue
        objetivos.append(
            {
                "id": fila["id"],
                "geometria": union_all(componentes) if componentes else from_wkb(fila["geometria"]),
                "seleccionado": fila["id"] == sitio,
                "provisional": provisional,
                "componentes": len(componentes),
            }
        )
    return sorted(objetivos, key=lambda o: (o["seleccionado"], o["id"]))


class VisorSeguimiento:
    """Adapta el visor del piloto, manteniendo Tkinter, Pillow y los recortes existentes."""

    def __init__(self, raiz, almacen, config: dict, limite: int = 12):
        self.raiz, self.almacen, self.config = raiz, almacen, config
        self.limite = limite
        self.anios = sorted((c["anio"] for c in config["campanias"]), reverse=True)
        self.historicos = [a for a in self.anios if a < 2023]
        self.edificios = gpd.read_file(
            ruta_proyecto("data/geografia/catastro/edificios_cantabria.gpkg"), fid_as_index=True
        )
        self.actual = None
        self.fotos = []
        self.zoom = 1.0
        self.centro = [0.5, 0.5]
        self.arrastre = None
        self.inicio_arrastre = None
        self.movimiento_arrastre = False
        self.marcas_objetivos = {}
        self.objetivos = []
        self.evento_borrador = None
        self.anio = tk.StringVar(value=str(self.historicos[0]))
        self.estado_referencia = tk.StringVar(value=NOMBRES["presente"])
        self.soporte = tk.StringVar()
        self.relacion = tk.StringVar()
        self.alcance = tk.StringVar(value="emplazamiento_completo")
        self.etiquetas = tk.StringVar()
        self.notas = tk.StringVar()
        self.cubiertas = tk.StringVar()
        self.capas = {
            k: tk.BooleanVar(value=k == "Agrupación")
            for k in ("Predicción", "Agrupación", "Catastro")
        }
        self.filtros = {
            k: tk.StringVar(value="Todos") for k in ("Municipio", "Estado", "Soporte", "Catastro")
        }
        self.solo_muestra = tk.BooleanVar(value=False)
        self.incluir_archivados = tk.BooleanVar(value=False)
        self.incluir_provisionales = tk.BooleanVar(value=False)
        self.umbral = tk.StringVar(value="0.5")
        self.ultimo_guardado = ""
        self.avanzadas = tk.BooleanVar(value=False)
        self.lote_terminado = False
        self.preparando = False
        self.seleccionando_candidato = False
        self.evento_preparacion = None
        self.mensajes_preparacion = Queue()
        self.anticipando = False
        self.evento_anticipacion = None
        self.mensajes_anticipacion = Queue()
        self.detener_anticipacion = Event()
        self.fallos_anticipacion = set()
        self._construir()
        self.recargar()
        self.cola = []
        self.renovar_cola()
        cursor = almacen.meta("cursor_revision", {})
        if cursor.get("sitio") in self.candidatas_preparadas():
            sitio = cursor["sitio"]
            if sitio not in self.cola:
                self.cola.insert(0, sitio)
            pendientes = [a for a in self.historicos if (sitio, a) not in self.observaciones]
            anio = cursor.get("anio")
            self.mostrar(sitio, anio if anio in pendientes else pendientes[0])
        else:
            self.siguiente(cargar=False)
        self.programar_reserva()

    def _construir(self):
        self.raiz.title("Revisión histórica fotovoltaica · Cantabria")
        self.raiz.geometry("1380x930")
        self.titulo = ttk.Label(self.raiz, font=("Segoe UI", 12), wraplength=1320)
        self.titulo.pack(fill="x", padx=10, pady=6)
        self.paso = ttk.Label(self.raiz, font=("Segoe UI", 13, "bold"))
        self.paso.pack(fill="x", padx=10)
        self.aviso = ttk.Label(self.raiz, wraplength=1320, foreground="#13623a")
        self.aviso.pack(fill="x", padx=10, pady=(2, 4))
        filtros = ttk.Frame(self.raiz)
        filtros.pack(fill="x", padx=10)
        for nombre, variable in self.filtros.items():
            ttk.Label(filtros, text=nombre).pack(side="left", padx=3)
            valores = (
                ("Todos", "Pendiente", "Revisado", "Especial", *DECISIONES)
                if nombre == "Estado"
                else (
                    ("Todos", *SOPORTES)
                    if nombre == "Soporte"
                    else ("Todos", *RELACIONES)
                    if nombre == "Catastro"
                    else ("Todos",)
                )
            )
            combo = ttk.Combobox(
                filtros, textvariable=variable, values=valores, state="readonly", width=20
            )
            combo.pack(side="left", padx=3)
            combo.bind("<<ComboboxSelected>>", lambda _: self.recargar())
            if nombre == "Municipio":
                self.selector_municipio = combo
        ttk.Checkbutton(
            filtros, text="Muestra previa", variable=self.solo_muestra, command=self.recargar
        ).pack(side="left")
        cuerpo = ttk.Panedwindow(self.raiz, orient="horizontal")
        cuerpo.pack(fill="both", expand=True, padx=10, pady=6)
        lista = ttk.Frame(cuerpo)
        self.tabla = ttk.Treeview(
            lista, columns=("municipio", "estado"), show="tree headings", height=18
        )
        self.tabla.heading("#0", text="Emplazamiento")
        self.tabla.heading("municipio", text="Municipio")
        self.tabla.heading("estado", text="Campaña")
        self.tabla.column("#0", width=140)
        self.tabla.column("municipio", width=110)
        self.tabla.column("estado", width=85)
        barra = ttk.Scrollbar(lista, command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=barra.set)
        self.tabla.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        self.tabla.bind("<<TreeviewSelect>>", self.seleccionar)
        cuerpo.add(lista, weight=1)
        derecha = ttk.Frame(cuerpo)
        cuerpo.add(derecha, weight=4)
        controles = ttk.Frame(derecha)
        controles.pack(fill="x")
        ttk.Label(controles, text="Observar campaña:").pack(side="left")
        for anio in self.historicos:
            ttk.Radiobutton(
                controles,
                text=str(anio),
                variable=self.anio,
                value=str(anio),
                command=self.cambiar_anio,
            ).pack(side="left")
        for nombre, variable in self.capas.items():
            ttk.Checkbutton(
                controles,
                text="Geometría original" if nombre == "Agrupación" else nombre,
                variable=variable,
                command=self.dibujar,
            ).pack(side="left", padx=4)
        ttk.Button(controles, text="Restablecer vista", command=self.restablecer).pack(side="left")
        ttk.Button(controles, text="Centrar objetivo", command=self.centrar_objetivo).pack(
            side="left"
        )
        self.indicacion_objetivo = ttk.Label(
            derecha, wraplength=1080, font=("Segoe UI", 10, "bold")
        )
        self.indicacion_objetivo.pack(fill="x", pady=3)
        imagenes = ttk.Frame(derecha)
        imagenes.pack(fill="both", expand=True)
        self.paneles = []
        for n in range(2):
            panel = tk.Canvas(
                imagenes, background="#17232e", highlightthickness=0, width=500, height=480
            )
            panel.pack(side="left", fill="both", expand=True, padx=3)
            panel.bind("<Configure>", lambda _: self.dibujar())
            panel.bind("<MouseWheel>", self.ampliar)
            panel.bind("<ButtonPress-1>", self.iniciar_arrastre)
            panel.bind("<B1-Motion>", self.desplazar)
            panel.bind("<ButtonRelease-1>", self.elegir_objetivo)
            self.paneles.append(panel)
        referencia = ttk.Frame(derecha)
        referencia.pack(fill="x", pady=3)
        ttk.Label(referencia, text="2023 (izquierda):").pack(side="left", padx=3)
        selector = ttk.Combobox(
            referencia,
            textvariable=self.estado_referencia,
            values=tuple(NOMBRES.values()),
            width=27,
            state="readonly",
        )
        selector.pack(side="left", padx=3)
        selector.bind("<<ComboboxSelected>>", lambda _: self.corregir_referencia())
        self.aviso_referencia = ttk.Label(referencia, wraplength=630)
        self.aviso_referencia.pack(side="left", padx=3)
        self.detalles = ttk.Label(derecha, wraplength=980)
        self.detalles.pack(fill="x", pady=4)
        campos = ttk.Frame(derecha)
        campos.pack(fill="x")
        for nombre, variable, valores in (
            ("Soporte físico", self.soporte, SOPORTES),
            ("Relación con Catastro", self.relacion, RELACIONES),
            (
                "Alcance",
                self.alcance,
                ("emplazamiento_completo", "zona_visible", "agrupacion_cubiertas"),
            ),
        ):
            ttk.Label(campos, text=nombre).pack(side="left", padx=3)
            combo = ttk.Combobox(
                campos, textvariable=variable, values=valores, width=24, state="readonly"
            )
            combo.pack(side="left", padx=3)
            combo.bind("<<ComboboxSelected>>", self.guardar_campos)
        entradas = ttk.Frame(derecha)
        entradas.pack(fill="x", pady=3)
        ttk.Label(entradas, text="Etiquetas (comas):").grid(row=0, column=0, sticky="w")
        ttk.Label(entradas, text="Notas de esta campaña:").grid(row=1, column=0, sticky="w")
        for fila, variable in enumerate((self.etiquetas, self.notas)):
            entrada = ttk.Entry(entradas, textvariable=variable, width=60)
            entrada.grid(row=fila, column=1, sticky="ew")
            entrada.bind("<KeyRelease>", self.borrador)
            entrada.bind("<FocusOut>", self.guardar_borrador)
            if fila == 0:
                menu = tk.Menu(entrada, tearoff=False)
                for etiqueta in ETIQUETAS:
                    menu.add_command(
                        label=etiqueta.replace("_", " "),
                        command=lambda t=etiqueta: self.anadir_etiqueta(t),
                    )
                entrada.bind("<Button-3>", lambda e, m=menu: m.tk_popup(e.x_root, e.y_root))
        entradas.columnconfigure(1, weight=1)
        self.menu_cambios = tk.Menu(entradas, tearoff=False)
        for etiqueta, nombre in CAMBIOS.items():
            self.menu_cambios.add_command(
                label=nombre, command=lambda t=etiqueta: self.marcar_cambio_referencia(t)
            )
        self.boton_cambios = ttk.Button(
            entradas, text="Cambio entre las fotos…", command=self.abrir_cambios
        )
        self.boton_cambios.grid(row=0, column=2, padx=4)
        botones = ttk.Frame(self.raiz)
        botones.pack(fill="x", padx=10, pady=4)
        self.botones_decision = {}
        for tecla, decision in zip("12345", DECISIONES):
            boton = ttk.Button(
                botones,
                text=f"{tecla} · {BOTONES[decision]}",
                command=lambda d=decision: self.decidir(d),
            )
            boton.pack(side="left", padx=3)
            self.botones_decision[decision] = boton
            self.raiz.bind(tecla, lambda e, d=decision: self.atajo(e, d))
        ttk.Button(botones, text="← Anterior / corregir", command=self.anterior).pack(
            side="left", padx=3
        )
        self.boton_siguiente = ttk.Button(botones, text="Siguiente →", command=self.siguiente)
        self.boton_siguiente.pack(side="left", padx=3)
        ttk.Button(botones, text="Deshacer", command=self.deshacer).pack(side="left", padx=3)
        acciones = ttk.Frame(self.raiz)
        acciones.pack(fill="x", padx=10, pady=3)
        ttk.Button(acciones, text="Confirmar agrupación", command=self.confirmar_grupo).pack(
            side="left", padx=3
        )
        self.boton_contexto = ttk.Button(acciones, text="Más contexto", command=self.contexto)
        self.boton_contexto.pack(side="left", padx=3)
        ttk.Button(acciones, text="Resumen", command=self.resumen).pack(side="left", padx=3)
        ttk.Button(acciones, text="Cómo revisar (F1)", command=self.ayuda).pack(side="left", padx=3)
        ttk.Checkbutton(
            acciones,
            text="Herramientas avanzadas",
            variable=self.avanzadas,
            command=self.mostrar_avanzadas,
        ).pack(side="left", padx=3)
        self.acciones_avanzadas = ttk.Frame(self.raiz)
        ttk.Button(self.acciones_avanzadas, text="Dividir / fusionar", command=self.reagrupar).pack(
            side="left", padx=3
        )
        ttk.Button(
            self.acciones_avanzadas, text="Corregir asociación / cubiertas", command=self.asociar
        ).pack(side="left", padx=3)
        ttk.Button(
            self.acciones_avanzadas, text="Añadir candidato histórico", command=self.anadir
        ).pack(side="left", padx=3)
        ttk.Button(
            self.acciones_avanzadas,
            text="Auditoría visual",
            command=lambda: webbrowser.open((SALIDA / "auditoria_visual.html").as_uri()),
        ).pack(side="left", padx=3)
        ttk.Checkbutton(
            self.acciones_avanzadas,
            text="Consultar originales archivados",
            variable=self.incluir_archivados,
            command=self.recargar,
        ).pack(side="left")
        ttk.Checkbutton(
            self.acciones_avanzadas,
            text="Mostrar zonas provisionales",
            variable=self.incluir_provisionales,
            command=self.recargar,
        ).pack(side="left")
        self.estado_carga = ttk.Label(self.raiz, foreground="#13623a", wraplength=1340)
        self.estado_carga.pack(fill="x", padx=10)
        self.progreso = ttk.Label(self.raiz, wraplength=1340)
        self.progreso.pack(fill="x", padx=10, pady=6)
        self.raiz.bind(
            "<Left>", lambda e: self.anterior() if not isinstance(e.widget, ttk.Entry) else None
        )
        self.raiz.bind(
            "<Right>", lambda e: self.siguiente() if not isinstance(e.widget, ttk.Entry) else None
        )
        self.raiz.bind("<Control-z>", lambda _: self.deshacer())
        self.raiz.bind("<F1>", lambda _: self.ayuda())
        self.raiz.bind("<Escape>", lambda _: self.cancelar_candidato())
        self.raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

    def mostrar_avanzadas(self):
        if self.avanzadas.get():
            self.acciones_avanzadas.pack(fill="x", padx=10, pady=3, before=self.progreso)
        else:
            self.acciones_avanzadas.pack_forget()
        self.dibujar()

    def ayuda(self):
        messagebox.showinfo(
            "Una respuesta por año",
            "El orden es 2020 → 2017 → 2014 → siguiente emplazamiento. "
            "2023 permanece a la izquierda; nunca se compara consigo mismo. "
            "Se saltan los años ya guardados. El zoom se conserva entre años.\n\n"
            "Las detecciones de una misma imagen van seguidas. Las siguientes fotos se "
            "preparan mientras respondes y se abren automáticamente. Si la descarga tarda "
            "más que tu revisión, verás su progreso sin que se bloquee la ventana.\n\n"
            "Si 2023 aún no está revisado, el primer clic confirma la respuesta indicada "
            "en 2023 (izquierda) y responde al histórico. Comprueba ambas fotos. "
            "Por defecto indica presencia clara; si no es correcta, cambia el selector "
            "de 2023 antes de responder. Los casos normales requieren tres respuestas.\n\n"
            "1: ves paneles. 2: ves la zona y no hay paneles.\n"
            "3: no sabes si son paneles. 4: la imagen no permite decidir.\n"
            "Un marco desplazado no exige ajustar máscaras: reconoce el mismo lugar en el "
            "entorno. Para marcar ausencia debes distinguir suficientemente la zona.\n"
            "5: no hay paneles en ninguna de las dos fotos. Guarda falso positivo de 2023 "
            "y ausencia en el histórico, y avanza. Si 2023 es falso pero antes sí había "
            "paneles, cambia solo el selector de 2023 y responde presencia al histórico. "
            "Para una referencia dudosa o no evaluable, usa ese selector.\n\n"
            "Si ya había paneles y después hay más, marca presencia en ambos años. "
            "En Cambio entre las fotos elige Más paneles en 2023. Se anota en 2023 "
            "sin volver a responder ese año. No tienes que contar módulos. "
            "Si no distingues bien el aumento, no lo marques.\n\n"
            "Suelo: presencia y soporte suelo. Tejado y suelo juntos: mixto. "
            "No cambies el soporte por estar fuera de Catastro.\n\n"
            "Confirma la agrupación una vez cuando hayas comprobado qué pertenece a este "
            "emplazamiento. Si mezcla casos distintos, déjala pendiente. "
            "Esto es independiente de responder a cada año.\n\n"
            "OBJETIVO naranja señala el caso evaluado en ambas fotos; OTRO CASO azul "
            "pertenece a otro emplazamiento. Haz clic en su marco para seleccionarlo. "
            "Las zonas provisionales sin detección quedan apartadas; puedes consultarlas "
            "desde Herramientas avanzadas → Mostrar zonas provisionales. "
            "El marco histórico señala la misma zona de 2023, no una predicción de ese año. "
            "Verde es solo la predicción. "
            "Usa Más contexto o alcance zona_visible si falta parte de la instalación. "
            "Puedes corregir con Anterior o deshacer con Ctrl+Z.",
            parent=self.raiz,
        )

    def recargar(self):
        self.sitios = {
            s["id"]: s
            for s in self.almacen.filas("SELECT * FROM emplazamientos ORDER BY municipio,id")
        }
        self.observaciones = {
            (o["emplazamiento"], o["anio"]): o
            for o in self.almacen.filas("SELECT * FROM observaciones")
        }
        localizados = {
            m["emplazamiento"]
            for m in self.almacen.filas(
                "SELECT DISTINCT m.emplazamiento FROM miembros m JOIN componentes c "
                "ON c.id=m.componente WHERE c.anio=2023"
            )
        }
        self.provisionales = {
            sid
            for sid, s in self.sitios.items()
            if s["origen"] in ("piloto", "inventario_2023") and sid not in localizados
        }
        self.selector_municipio["values"] = (
            "Todos",
            *sorted({s["municipio"] for s in self.sitios.values()}),
        )
        self.tabla.delete(*self.tabla.get_children())
        anio = int(self.anio.get())
        muestra = self.almacen.meta("muestra_auditoria", [])
        muestra_sitios = {
            m["emplazamiento"]
            for m in self.almacen.filas("SELECT * FROM miembros")
            if m["componente"] in muestra
        }
        muestra_sitios.update(self.almacen.meta("muestra_previa_sitios", []))
        self.visibles = []
        self.filtrados = []
        self.para_preparar = []
        for sid, s in self.sitios.items():
            if not s["activo"] and not self.incluir_archivados.get():
                continue
            if self.solo_muestra.get() and sid not in muestra_sitios:
                continue
            if any(
                self.filtros[n].get() not in ("Todos", s[c])
                for n, c in (
                    ("Municipio", "municipio"),
                    ("Soporte", "soporte"),
                    ("Catastro", "relacion"),
                )
            ):
                continue
            self.para_preparar.append(sid)
            if sid in self.provisionales and not self.incluir_provisionales.get():
                continue
            self.filtrados.append(sid)
            decision = self.observaciones.get((sid, anio), {}).get("decision", "Pendiente")
            estado = self.filtros["Estado"].get()
            if (
                estado == "Pendiente"
                and decision != "Pendiente"
                or estado == "Revisado"
                and decision == "Pendiente"
            ):
                continue
            if estado == "referencia_incorrecta":
                if self.observaciones.get((sid, 2023), {}).get("decision") != estado:
                    continue
            elif estado in DECISIONES and decision != estado:
                continue
            if estado == "Especial" and not (
                json.loads(s["etiquetas"]) or decision in ("dudosa", "no_evaluable")
            ):
                continue
            self.visibles.append(sid)
            self.tabla.insert("", "end", iid=sid, text=sid, values=(s["municipio"], decision))
        principales = {
            sid for sid, s in self.sitios.items() if s["activo"] and sid not in self.provisionales
        }
        total = len(principales) * len(self.historicos)
        hechas = sum(s in principales for s, a in self.observaciones if a in self.historicos)
        referencias = sum(self.sitios[s]["activo"] for s, a in self.observaciones if a == 2023)
        self.progreso.config(
            text=f"Históricos localizados: {hechas}/{total} guardados ({100 * hechas / total if total else 0:.1f} %). Pendientes: {total - hechas}. Referencias 2023 revisadas: {referencias}. "
            f"Zonas provisionales apartadas: {sum(self.sitios[s]['activo'] for s in self.provisionales)}. "
            "OBJETIVO naranja; otros casos azules; Catastro amarillo opcional. Rueda: ampliar. Arrastrar: desplazar ambas fotos. "
            "La ausencia de predicción no significa ausencia de paneles. La recuperación regional sigue pendiente."
        )
        if self.lote_terminado and not self.preparando:
            self.anunciar_fin_lote()
        if hasattr(self, "cola"):
            self.programar_reserva()

    def candidatas_pendientes(self):
        """Selecciona casos activos con históricos pendientes respetando los filtros."""
        visibles = self.filtrados if self.filtros["Estado"].get() == "Pendiente" else self.visibles
        muestra = self.almacen.meta("muestra_auditoria", [])
        preferidos = {
            m["emplazamiento"]
            for m in self.almacen.filas("SELECT * FROM miembros")
            if m["componente"] in muestra
        }
        preferidos.update(self.almacen.meta("muestra_previa_sitios", []))
        candidatas = [
            s
            for s in visibles
            if self.sitios[s]["activo"]
            and any((s, a) not in self.observaciones for a in self.historicos)
        ]
        ventanas_preferidas = {self.sitios[s]["ventana"] for s in preferidos if s in self.sitios}
        return sorted(
            candidatas,
            key=lambda s: (
                self.sitios[s]["ventana"] not in ventanas_preferidas,
                self.sitios[s]["ventana"],
                s,
            ),
        )

    def candidatas_preparadas(self):
        """Reutiliza fotos existentes; los errores conservados se pueden marcar no evaluables."""
        recursos = {
            (r["ventana"], r["anio"]): r
            for r in self.almacen.filas("SELECT ventana,anio,imagen,error FROM recursos")
        }
        return [
            s
            for s in self.candidatas_pendientes()
            if all(
                r and (r["imagen"] and Path(r["imagen"]).exists() or r["error"])
                for a in self.anios
                for r in (recursos.get((self.sitios[s]["ventana"], a)),)
            )
        ]

    def renovar_cola(self):
        """Continúa otro lote preparado sin repetir casos terminados ni descargar fotos."""
        candidatas = self.candidatas_preparadas()
        ventanas = set(
            list(dict.fromkeys(self.sitios[s]["ventana"] for s in candidatas))[: self.limite]
        )
        self.cola = [s for s in candidatas if self.sitios[s]["ventana"] in ventanas]
        return bool(self.cola)

    def sitios_sin_fotos(self, incluir_fallidos=True):
        """Incluye ubicaciones sin procesar para descubrir objetivos, sin revisar negativos."""
        recursos = {
            (r["ventana"], r["anio"]): r for r in self.almacen.filas("SELECT * FROM recursos")
        }
        candidatas = set(self.candidatas_pendientes())
        if self.filtros["Estado"].get() in ("Todos", "Pendiente"):
            candidatas.update(self.para_preparar)
        pendientes = []
        for sid in sorted(candidatas, key=lambda s: (self.sitios[s]["ventana"], s)):
            sitio = self.sitios[sid]
            ventana = sitio["ventana"]
            if not sitio["activo"] or all((sid, a) in self.observaciones for a in self.historicos):
                continue
            if not incluir_fallidos and ventana in self.fallos_anticipacion:
                continue
            referencia = recursos.get((ventana, 2023), {})
            if sid in self.provisionales and not self.incluir_provisionales.get():
                procesada = referencia.get("error") or (
                    referencia.get("prediccion") and Path(referencia["prediccion"]).exists()
                )
                if procesada:
                    continue
            # Un legado sin componentes necesita localizar primero el objetivo de 2023.
            if (sid not in self.provisionales or referencia.get("prediccion")) and all(
                r and (r["imagen"] and Path(r["imagen"]).exists() or r["error"])
                for a in self.anios
                for r in (recursos.get((ventana, a)),)
            ):
                continue
            pendientes.append(sid)
        return pendientes

    def programar_reserva(self):
        if (
            self.config.get("precarga", True)
            and not self.anticipando
            and not self.detener_anticipacion.is_set()
            and self.evento_anticipacion is None
        ):
            self.evento_anticipacion = self.raiz.after(200, self.mantener_reserva)

    def mantener_reserva(self):
        """Prepara un máximo de un lote por delante sin bloquear la revisión actual."""
        self.evento_anticipacion = None
        if self.detener_anticipacion.is_set() or self.anticipando or self.preparando:
            return
        actual = self.sitios[self.actual]["ventana"] if self.actual else None
        listas = {self.sitios[s]["ventana"] for s in self.candidatas_preparadas()}
        reserva = len(listas - {actual})
        capacidad = self.limite - reserva
        pendientes = self.sitios_sin_fotos(incluir_fallidos=False)
        ventanas = list(dict.fromkeys(self.sitios[s]["ventana"] for s in pendientes))[
            : max(0, capacidad)
        ]
        sitios = [s for s in pendientes if self.sitios[s]["ventana"] in ventanas]
        if not sitios:
            if self.lote_terminado and self.candidatas_preparadas():
                self.siguiente(cargar=False)
            elif self.lote_terminado:
                self.anunciar_fin_lote()
            return
        self.anticipando = True
        self.ventanas_anticipadas = set(ventanas)
        self.estado_carga.config(
            text=f"Preparando {len(ventanas)} imágenes por delante. Puedes seguir respondiendo."
        )
        if self.lote_terminado:
            self.anunciar_fin_lote()
        ruta, config = self.almacen.ruta, self.config
        mensajes, detener = self.mensajes_anticipacion, self.detener_anticipacion
        solo_localizados = not self.incluir_provisionales.get()

        def anticipar():
            try:
                from paneles_solares.etiquetado.preparar_seguimiento import (
                    preparar_lote,
                )

                almacen = AlmacenSeguimiento(ruta)
                try:
                    preparar_lote(
                        almacen,
                        config,
                        len(ventanas),
                        sitios,
                        progreso=lambda t: mensajes.put(("estado", t)),
                        al_completar=lambda v: mensajes.put(("ventana", v)),
                        solo_localizados=solo_localizados,
                        detener=detener,
                    )
                finally:
                    almacen.cerrar()
            except Exception as error:  # noqa: BLE001 -- Recuperar el control sin perder respuestas.
                mensajes.put(("error", str(error)))
            else:
                mensajes.put(("fin", None))

        self.hilo_anticipacion = Thread(target=anticipar, daemon=True)
        self.hilo_anticipacion.start()
        self.evento_anticipacion = self.raiz.after(100, self.comprobar_anticipacion)

    def comprobar_anticipacion(self):
        """Publica ventanas completas sin mover el caso que el usuario está revisando."""
        self.evento_anticipacion = None
        refrescar, terminado, error = False, False, None
        try:
            while True:
                tipo, texto = self.mensajes_anticipacion.get_nowait()
                if tipo == "estado":
                    self.estado_carga.config(text=f"Preparando siguientes fotos: {texto}")
                elif tipo == "ventana":
                    refrescar = True
                else:
                    terminado, refrescar = True, True
                    if tipo == "error":
                        error = texto
        except Empty:
            pass
        if terminado:
            self.anticipando = False
        if refrescar:
            self.recargar()
            if terminado:
                incompletas = {self.sitios[s]["ventana"] for s in self.sitios_sin_fotos()}
                self.fallos_anticipacion.update(self.ventanas_anticipadas & incompletas)
                self.estado_carga.config(
                    text=(
                        f"Carga pendiente: {error}. Puedes seguir con las fotos disponibles; Siguiente reintenta al llegar al final."
                        if error
                        else "Siguientes fotos preparadas."
                    )
                )
            if self.lote_terminado and not self.preparando:
                self.siguiente(cargar=False)
            elif self.actual and self.tabla.exists(self.actual):
                self.tabla.selection_set(self.actual)
                self.tabla.see(self.actual)
        if self.anticipando:
            self.evento_anticipacion = self.raiz.after(100, self.comprobar_anticipacion)
        else:
            self.programar_reserva()

    def seleccionar(self, _):
        seleccion = self.tabla.selection()
        if seleccion and (self.actual is None or seleccion[0] != self.actual):
            sitio = seleccion[0]
            pendientes = [a for a in self.historicos if (sitio, a) not in self.observaciones]
            self.mostrar(sitio, pendientes[0] if pendientes else int(self.anio.get()))

    def mostrar(self, sitio, anio, guardar_actual=True):
        self.seleccionando_candidato = False
        if guardar_actual:
            self.guardar_borrador()
        cambio_sitio = self.actual != sitio
        if anio == 2023:
            pendientes = [a for a in self.historicos if (sitio, a) not in self.observaciones]
            anio = pendientes[0] if pendientes else self.historicos[0]
        self.lote_terminado = False
        self.boton_siguiente.config(text="Siguiente →")
        self.boton_siguiente.state(["disabled"] if self.preparando else ["!disabled"])
        self.actual = sitio
        self.anio.set(str(anio))
        if cambio_sitio:
            self.zoom, self.centro = 1.0, [0.5, 0.5]
        self.inicio = time.monotonic()
        s = self.sitios[sitio]
        self.soporte.set(s["soporte"])
        self.relacion.set(s["relacion"])
        o = self.observaciones.get((sitio, anio), {})
        borrador = self.almacen.meta(f"borrador_{sitio}_{anio}", {})
        self.etiquetas.set(
            borrador.get("etiquetas", ", ".join(json.loads(o.get("etiquetas", "[]"))))
        )
        self.notas.set(borrador.get("notas", o.get("notas", "")))
        self.alcance.set(
            borrador.get(
                "alcance",
                o.get(
                    "alcance",
                    self.alcance_inicial(sitio),
                ),
            )
        )
        self.almacen.poner_meta("cursor_revision", {"sitio": sitio, "anio": anio})
        self.recargar()
        if self.tabla.exists(sitio):
            self.tabla.selection_set(sitio)
            self.tabla.see(sitio)
        self.dibujar()
        if self.config.get("precarga", True) and not self.anticipando:
            self.programar_reserva()

    def alcance_inicial(self, sitio):
        """Distingue paneles cortados del antiguo aviso de edificio fuera del recorte."""
        componentes = self.almacen.filas(
            "SELECT c.borde,c.geometria FROM componentes c JOIN miembros m ON m.componente=c.id WHERE m.emplazamiento=? AND c.anio=2023",
            (sitio,),
        )
        parcial = (
            any(c["borde"] for c in componentes)
            if componentes
            else "borde_recorte" in json.loads(self.sitios[sitio]["etiquetas"])
        )
        if parcial:
            return "zona_visible"
        geometria = (
            union_all([from_wkb(c["geometria"]) for c in componentes])
            if componentes
            else from_wkb(self.sitios[sitio]["geometria"])
        )
        recursos = self._recursos_sitio()
        for anio in (2023, int(self.anio.get())):
            cobertura = union_all(
                [
                    box(r["xmin"], r["ymin"], r["xmax"], r["ymax"])
                    for r in recursos
                    if r["anio"] == anio and r["imagen"] and Path(r["imagen"]).exists()
                ]
            )
            if not cobertura.covers(geometria):
                return "zona_visible"
        return "emplazamiento_completo"

    def cambiar_anio(self):
        # El borrador del año anterior ya quedó guardado después de cada edición.
        if self.actual:
            self.mostrar(self.actual, int(self.anio.get()), guardar_actual=False)

    def siguiente(self, tras_respuesta=False, cargar=True):
        if self.preparando:
            return
        self.cancelar_candidato()
        # El filtro del año respondido puede ocultar el sitio recién guardado.
        if tras_respuesta and self.actual:
            for anio in self.historicos:
                if (self.actual, anio) not in self.observaciones:
                    self.mostrar(self.actual, anio)
                    return
        # Termina los históricos de cada sitio antes de pasar al siguiente.
        visibles = self.filtrados if self.filtros["Estado"].get() == "Pendiente" else self.visibles
        orden = [s for s in self.cola if s in visibles]
        if self.actual:
            ventana = self.sitios[self.actual]["ventana"]
            vecinos = [
                s for s in self.candidatas_preparadas() if self.sitios[s]["ventana"] == ventana
            ]
            orden = (
                ([self.actual] if self.actual in orden else [])
                + [s for s in vecinos if s != self.actual]
                + [s for s in orden if self.sitios[s]["ventana"] != ventana]
            )
        pendientes = [
            (sitio, anio)
            for sitio in orden
            for anio in self.historicos
            if (sitio, anio) not in self.observaciones
        ]
        actual = (self.actual, int(self.anio.get()))
        if actual in pendientes and len(pendientes) > 1:
            posicion = pendientes.index(actual)
            pendientes = pendientes[posicion + 1 :] + pendientes[: posicion + 1]
        if pendientes:
            self.mostrar(*pendientes[0])
            return
        if self.renovar_cola():
            sitio = self.cola[0]
            anio = next(a for a in self.historicos if (sitio, a) not in self.observaciones)
            self.mostrar(sitio, anio)
            return
        self.lote_terminado = True
        self.anunciar_fin_lote()
        self.aviso.config(text=self.ultimo_guardado)
        for boton in self.botones_decision.values():
            boton.state(["disabled"])
        if self.config.get("precarga", True):
            if cargar and not tras_respuesta:
                self.fallos_anticipacion.clear()
            self.programar_reserva()
            return
        if cargar and not tras_respuesta and self.candidatas_pendientes():
            self.cargar_siguiente_lote()

    def anunciar_fin_lote(self):
        if self.config.get("precarga", True) and (self.anticipando or self.sitios_sin_fotos()):
            texto = "Preparando las siguientes fotos… Se abrirán automáticamente; tus respuestas están guardadas."
            self.boton_siguiente.config(
                text="Preparando…" if self.anticipando else "Reintentar carga →"
            )
            self.boton_siguiente.state(["disabled"] if self.anticipando else ["!disabled"])
            self.paso.config(text=texto)
            return
        if self.candidatas_pendientes():
            texto = "Lote guardado. Pulsa «Continuar revisión →» para cargar las siguientes fotos."
            self.boton_siguiente.config(text="Continuar revisión →")
            self.boton_siguiente.state(["!disabled"])
        elif any(
            s["activo"]
            and (sid not in self.provisionales or self.incluir_provisionales.get())
            and any((sid, a) not in self.observaciones for a in self.historicos)
            for sid, s in self.sitios.items()
        ):
            texto = "No quedan casos pendientes con estos filtros. Cambia los filtros para seguir."
            self.boton_siguiente.state(["disabled"])
        else:
            texto = "Revisión de los casos localizados terminada. Las zonas provisionales siguen apartadas."
            self.boton_siguiente.state(["disabled"])
        self.paso.config(text=texto)

    def cargar_siguiente_lote(self):
        """Prepara un lote pequeño en segundo plano y abre su primer histórico pendiente."""
        self.guardar_borrador()
        candidatas = self.candidatas_pendientes()
        ventanas = list(dict.fromkeys(self.sitios[s]["ventana"] for s in candidatas))[: self.limite]
        sitios = [s for s in candidatas if self.sitios[s]["ventana"] in ventanas]
        if not sitios:
            return

        def preparar(almacen, informar):
            from paneles_solares.etiquetado.preparar_seguimiento import preparar_lote

            preparar_lote(almacen, self.config, self.limite, sitios, progreso=informar)

        self.iniciar_carga(preparar, "Cargando siguientes fotos…")

    def iniciar_carga(self, tarea, descripcion, conservar_caso=False):
        """Mantiene el visor disponible durante descargas y conserva cada avance en SQLite."""
        self.guardar_borrador()
        self.preparando = True
        self.conservar_caso = conservar_caso
        self.descripcion_carga = descripcion
        self.paso.config(text=f"{descripcion} Tus respuestas están guardadas.")
        self.boton_siguiente.config(text="Cargando…")
        self.boton_siguiente.state(["disabled"])
        self.boton_contexto.state(["disabled"])
        for boton in self.botones_decision.values():
            boton.state(["disabled"])

        def preparar():
            try:
                almacen = AlmacenSeguimiento(self.almacen.ruta)
                try:
                    tarea(almacen, lambda texto: self.mensajes_preparacion.put(("estado", texto)))
                finally:
                    almacen.cerrar()
            except Exception as error:  # noqa: BLE001 -- El visor debe recuperar el control del hilo.
                self.mensajes_preparacion.put(("error", str(error)))
            else:
                self.mensajes_preparacion.put(("fin", None))

        Thread(target=preparar, daemon=True).start()
        self.evento_preparacion = self.raiz.after(100, self.comprobar_preparacion)

    def comprobar_preparacion(self):
        self.evento_preparacion = None
        try:
            while True:
                tipo, texto = self.mensajes_preparacion.get_nowait()
                if tipo == "estado":
                    self.paso.config(text=f"{self.descripcion_carga} {texto}")
                    continue
                self.preparando = False
                self.boton_contexto.state(["!disabled"])
                self.recargar()
                if self.conservar_caso:
                    self.boton_siguiente.config(text="Siguiente →")
                    self.boton_siguiente.state(["!disabled"])
                    self.dibujar()
                    self.centrar_objetivo()
                    if self.lote_terminado:
                        self.anunciar_fin_lote()
                else:
                    self.siguiente(cargar=False)
                if tipo == "error":
                    messagebox.showerror("No se pudieron cargar las fotos", texto, parent=self.raiz)
                self.raiz.focus_set()
                self.programar_reserva()
                return
        except Empty:
            self.evento_preparacion = self.raiz.after(100, self.comprobar_preparacion)

    def anterior(self):
        acciones = self.almacen.filas(
            "SELECT objetivo,anio FROM acciones WHERE tipo IN ('observacion','comparacion') AND deshecha=0 ORDER BY id DESC"
        )
        for a in acciones:
            if (a["objetivo"], a["anio"]) != (self.actual, int(self.anio.get())) and a[
                "objetivo"
            ] in self.sitios:
                self.mostrar(a["objetivo"], a["anio"])
                return
        if self.actual in self.visibles:
            self.mostrar(
                self.visibles[max(0, self.visibles.index(self.actual) - 1)], int(self.anio.get())
            )

    def _recursos_sitio(self):
        s = self.sitios[self.actual]
        ids = {s["ventana"]}
        ids.update(
            c["ventana"]
            for c in self.almacen.filas(
                "SELECT ventana FROM contextos WHERE emplazamiento=?", (self.actual,)
            )
        )
        ids.update(
            c["ventana"]
            for c in self.almacen.filas(
                "SELECT c.ventana FROM componentes c JOIN miembros m ON c.id=m.componente WHERE m.emplazamiento=?",
                (self.actual,),
            )
        )
        return self.almacen.filas(
            "SELECT r.*,v.xmin,v.ymin,v.xmax,v.ymax FROM recursos r JOIN ventanas v ON r.ventana=v.id WHERE r.ventana IN ("
            + ",".join("?" for _ in ids)
            + ")",
            tuple(ids),
        )

    def dibujar(self):
        if self.actual is None:
            return
        s = self.sitios[self.actual]
        anio = int(self.anio.get())
        o = self.observaciones.get((self.actual, anio), {})
        self.titulo.config(
            text=f"{self.actual} · {s['municipio']} · OBSERVAR {anio} · {NOMBRES.get(o.get('decision'), 'Pendiente')} · agrupación {s['agrupacion']}"
        )
        ref = self.observaciones.get((self.actual, 2023))
        recursos = self._recursos_sitio()
        foto_ref = any(
            r["anio"] == 2023 and r["imagen"] and Path(r["imagen"]).exists() for r in recursos
        )
        provisional = (
            s["origen"] in ("piloto", "inventario_2023")
            and not self.almacen.db.execute(
                "SELECT 1 FROM miembros m JOIN componentes c ON c.id=m.componente WHERE m.emplazamiento=? AND c.anio=2023",
                (self.actual,),
            ).fetchone()
        )
        estado_ref = (
            ref["decision"]
            if ref
            else "no_evaluable"
            if not foto_ref
            else "dudosa"
            if s["origen"] == "candidato_manual" or provisional
            else "presente"
        )
        self.estado_referencia.set(NOMBRES[estado_ref])
        self.aviso_referencia.config(
            text="Guardado. Cámbialo solo si necesitas corregirlo."
            if ref
            else "El primer clic confirma esta respuesta de 2023; cámbiala si no coincide con la foto."
        )
        if not self.lote_terminado and not self.preparando and not self.seleccionando_candidato:
            texto = f"Responde a {anio}: ¿hay paneles? · Un clic guarda y pasa al año anterior"
            if ref is None:
                texto = f"Este clic guarda 2023: {NOMBRES[estado_ref]} + tu respuesta a {anio}"
            self.paso.config(text=texto)
        estados = " · ".join(
            f"{a}: {NOMBRES.get(self.observaciones.get((self.actual, a), {}).get('decision'), 'pendiente')}"
            for a in self.anios
        )
        self.aviso.config(text=f"{self.ultimo_guardado}\n{estados}".strip())
        for boton in self.botones_decision.values():
            boton.state(
                ["disabled"]
                if self.lote_terminado or self.preparando or self.seleccionando_candidato
                else ["!disabled"]
            )
        self.boton_cambios.state(["disabled"] if anio == 2023 else ["!disabled"])
        self.detalles.config(
            text="Para los recuentos, indica el soporte y confirma la agrupación una vez por caso si está clara. "
            "Si la instalación sigue fuera de la foto, usa Más contexto o guarda alcance zona_visible. "
            "5: no hay paneles en ambas fotos. Para corregir solo 2023, usa su selector."
            + (
                " Estás revisando zona_visible; si ya ves todo el emplazamiento, cambia Alcance a emplazamiento_completo."
                if self.alcance.get() == "zona_visible"
                else ""
            )
            + (
                f" Referencias Catastro: {s['edificios']}; cubiertas: {s['cubiertas']}; avisos: {s['etiquetas']}."
                if self.avanzadas.get()
                else ""
            )
            + (
                " Sugerencia por posición: cubierta catastrada; compruébala en la foto antes de cambiar el soporte."
                if s["soporte"] == "desconocido" and s["relacion"] == "interseccion_clara"
                else ""
            )
        )
        if not recursos:
            return
        xmin, ymin = min(r["xmin"] for r in recursos), min(r["ymin"] for r in recursos)
        xmax, ymax = max(r["xmax"] for r in recursos), max(r["ymax"] for r in recursos)
        lado_m = max(xmax - xmin, ymax - ymin)
        centro_x, centro_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        xmin, xmax = centro_x - lado_m / 2, centro_x + lado_m / 2
        ymin, ymax = centro_y - lado_m / 2, centro_y + lado_m / 2
        extension = (xmin, ymin, xmax, ymax)
        self.extension_vista = extension
        self.objetivos = objetivos_visuales(
            self.almacen, self.actual, sorted({r["ventana"] for r in recursos})
        )
        objetivo = next(o for o in self.objetivos if o["seleccionado"])
        vecinos = sum(not o["seleccionado"] for o in self.objetivos)
        self.indicacion_objetivo.config(
            text=(
                "UBICACIÓN PROVISIONAL: este caso no tiene detección de 2023 asociada. "
                "El contorno gris procede de la geometría antigua. "
                if objetivo["provisional"]
                else "Evalúa el emplazamiento señalado como OBJETIVO en ambas fotos. "
            )
            + (
                f"Hay {vecinos} otro(s) caso(s) en azul; haz clic en su marco para seleccionarlo."
                if vecinos
                else ""
            )
        )
        self.fotos = []
        for panel, campania in zip(self.paneles, (2023, anio)):
            base = Image.new("RGB", (900, 900), "#17232e")
            disponibles = [
                r
                for r in recursos
                if r["anio"] == campania and r["imagen"] and Path(r["imagen"]).exists()
            ]
            for r in disponibles:
                imagen = Image.open(r["imagen"]).convert("RGB")
                w = max(1, round((r["xmax"] - r["xmin"]) / (xmax - xmin) * 900))
                h = max(1, round((r["ymax"] - r["ymin"]) / (ymax - ymin) * 900))
                if (
                    self.capas["Predicción"].get()
                    and r["prediccion"]
                    and Path(r["prediccion"]).exists()
                ):
                    mascara = np.asarray(Image.open(r["prediccion"])) >= 32768
                    capa = Image.fromarray(mascara.astype("uint8") * 100).resize(imagen.size)
                    imagen = Image.composite(Image.new("RGB", imagen.size, "#35ff93"), imagen, capa)
                base.paste(
                    imagen.resize((w, h), Image.Resampling.BILINEAR),
                    (
                        round((r["xmin"] - xmin) / (xmax - xmin) * 900),
                        round((ymax - r["ymax"]) / (ymax - ymin) * 900),
                    ),
                )
            dibujo = ImageDraw.Draw(base)
            if self.capas["Catastro"].get():
                for g in self.edificios.geometry.iloc[
                    self.edificios.sindex.query(box(*extension), predicate="intersects")
                ]:
                    trazar_geometria(dibujo, g, extension, base.size, "#ffd740", 2)
            if self.capas["Agrupación"].get():
                trazar_geometria(
                    dibujo, from_wkb(s["geometria"]), extension, base.size, "#9da8b2", 2
                )
                componentes = self.almacen.filas(
                    "SELECT c.* FROM componentes c JOIN miembros m ON c.id=m.componente WHERE m.emplazamiento=?",
                    (self.actual,),
                )
                for c in componentes:
                    trazar_geometria(
                        dibujo, from_wkb(c["geometria"]), extension, base.size, "#ff923e", 3
                    )
            ancho, alto = max(panel.winfo_width(), 100), max(panel.winfo_height(), 100)
            lado = min(ancho, alto - 30)
            paso = 900 / self.zoom
            cx, cy = self.centro[0] * 900, self.centro[1] * 900
            foto = ImageTk.PhotoImage(
                base.crop((cx - paso / 2, cy - paso / 2, cx + paso / 2, cy + paso / 2)).resize(
                    (lado, lado), Image.Resampling.BILINEAR
                )
            )
            self.fotos.append(foto)
            panel.delete("all")
            panel.create_image(ancho / 2, (alto + 30) / 2, image=foto)
            self.marcar_objetivos(panel, extension, paso, cx, cy, lado, ancho, alto)
            panel.create_text(
                12,
                14,
                text=f"{campania} · {'RESPONDE A ESTA FOTO' if panel == self.paneles[1] else 'Referencia para comparar'}"
                + (" · IMAGEN NO DISPONIBLE" if not disponibles else ""),
                fill="white",
                anchor="w",
                font=("Segoe UI", 11),
            )

    def marcar_objetivos(self, panel, extension, paso, cx, cy, lado, ancho, alto):
        """Dibuja marcos y rótulos constantes, compartiendo coordenadas entre campañas."""
        xmin, ymin, xmax, ymax = extension
        izquierda, arriba = (ancho - lado) / 2, (alto + 30 - lado) / 2
        visibles = box(
            xmin + (cx - paso / 2) / 900 * (xmax - xmin),
            ymax - (cy + paso / 2) / 900 * (ymax - ymin),
            xmin + (cx + paso / 2) / 900 * (xmax - xmin),
            ymax - (cy - paso / 2) / 900 * (ymax - ymin),
        )
        marcas = []
        for objetivo in self.objetivos:
            geometria = objetivo["geometria"].intersection(visibles)
            if geometria.is_empty:
                continue
            partes = geometria.geoms if hasattr(geometria, "geoms") else [geometria]
            for indice, parte in enumerate(sorted(partes, key=lambda g: g.area, reverse=True)):
                x0, y0, x1, y1 = parte.bounds
                x0 = izquierda + ((x0 - xmin) / (xmax - xmin) * 900 - cx + paso / 2) / paso * lado
                x1 = izquierda + ((x1 - xmin) / (xmax - xmin) * 900 - cx + paso / 2) / paso * lado
                top = arriba + ((ymax - y1) / (ymax - ymin) * 900 - cy + paso / 2) / paso * lado
                bottom = arriba + ((ymax - y0) / (ymax - ymin) * 900 - cy + paso / 2) / paso * lado
                x0, x1 = max(izquierda, x0 - 5), min(izquierda + lado, x1 + 5)
                top, bottom = max(arriba, top - 5), min(arriba + lado, bottom + 5)
                seleccionado = objetivo["seleccionado"]
                provisional = objetivo["provisional"]
                color = "#9da8b2" if provisional else "#ff923e" if seleccionado else "#40d9ff"
                etiqueta = (
                    "ZONA PROVISIONAL"
                    if provisional
                    else "OBJETIVO"
                    if seleccionado
                    else "OTRO CASO"
                )
                tags = ("localizador", "objetivo" if seleccionado else "otro_caso", objetivo["id"])
                panel.create_rectangle(
                    x0,
                    top,
                    x1,
                    bottom,
                    outline=color,
                    width=3,
                    dash=(5, 3) if provisional else (),
                    tags=tags,
                )
                rotulo = None
                etiqueta_x = max(izquierda + 2, min(x0 + 3, izquierda + lado - 120))
                etiqueta_y = max(arriba + 4, top - 20)
                if indice == 0:
                    texto = panel.create_text(
                        etiqueta_x + 4,
                        etiqueta_y + 3,
                        text=etiqueta,
                        fill=color,
                        anchor="nw",
                        font=("Segoe UI", 10, "bold"),
                        tags=tags,
                    )
                    fondo = panel.create_rectangle(
                        panel.bbox(texto), fill="#17232e", outline=color, tags=tags
                    )
                    panel.tag_raise(texto, fondo)
                    rotulo = panel.bbox(texto)
                marcas.append(
                    {
                        "id": objetivo["id"],
                        "seleccionado": seleccionado,
                        "rectangulo": (x0, top, x1, bottom),
                        "rotulo": rotulo,
                    }
                )
        self.marcas_objetivos[panel] = marcas
        if not any(m["seleccionado"] for m in marcas):
            panel.create_text(
                12,
                42,
                text="OBJETIVO FUERA DE VISTA · pulsa Centrar objetivo",
                fill="#ff923e",
                anchor="w",
                font=("Segoe UI", 10, "bold"),
                tags=("localizador",),
            )

    def centrar_objetivo(self):
        if not self.actual or not self.objetivos:
            return
        objetivo = next(o for o in self.objetivos if o["seleccionado"])
        xmin, ymin, xmax, ymax = self.extension_vista
        geometria = objetivo["geometria"].intersection(box(xmin, ymin, xmax, ymax))
        if geometria.is_empty:
            self.restablecer()
            return
        x0, y0, x1, y1 = geometria.bounds
        self.zoom = min(6.0, max(1.0, (xmax - xmin) / (max(x1 - x0, y1 - y0) + 20)))
        margen = 0.5 / self.zoom
        self.centro = [
            min(1 - margen, max(margen, ((x0 + x1) / 2 - xmin) / (xmax - xmin))),
            min(1 - margen, max(margen, (ymax - (y0 + y1) / 2) / (ymax - ymin))),
        ]
        self.dibujar()

    def iniciar_arrastre(self, evento):
        self.arrastre = self.inicio_arrastre = (evento.x, evento.y)
        self.movimiento_arrastre = False

    def elegir_objetivo(self, evento):
        self.arrastre = None
        if self.movimiento_arrastre:
            return
        if self.seleccionando_candidato:
            self.crear_candidato_en_foto(evento)
            return
        for marca in reversed(self.marcas_objetivos.get(evento.widget, [])):
            if marca["seleccionado"]:
                continue
            zonas = [marca["rectangulo"]]
            if marca["rotulo"]:
                zonas.append(marca["rotulo"])
            if any(x0 <= evento.x <= x1 and y0 <= evento.y <= y1 for x0, y0, x1, y1 in zonas):
                sitio = marca["id"]
                if sitio not in self.cola:
                    posicion = self.cola.index(self.actual) + 1 if self.actual in self.cola else 0
                    self.cola.insert(posicion, sitio)
                pendientes = [a for a in self.historicos if (sitio, a) not in self.observaciones]
                self.mostrar(sitio, pendientes[0] if pendientes else int(self.anio.get()))
                self.raiz.focus_set()
                return

    def ampliar(self, evento):
        self.zoom = min(8.0, max(1.0, self.zoom * (1.25 if evento.delta > 0 else 0.8)))
        self.dibujar()

    def desplazar(self, evento):
        if self.arrastre:
            if (
                self.inicio_arrastre
                and max(
                    abs(evento.x - self.inicio_arrastre[0]), abs(evento.y - self.inicio_arrastre[1])
                )
                > 4
            ):
                self.movimiento_arrastre = True
            lado = max(100, min(evento.widget.winfo_width(), evento.widget.winfo_height() - 30))
            self.centro = [
                self.centro[0] - (evento.x - self.arrastre[0]) / lado / self.zoom,
                self.centro[1] - (evento.y - self.arrastre[1]) / lado / self.zoom,
            ]
            self.arrastre = (evento.x, evento.y)
            self.dibujar()

    def restablecer(self):
        self.zoom, self.centro = 1.0, [0.5, 0.5]
        self.dibujar()

    def atajo(self, evento, decision):
        if isinstance(evento.widget, (ttk.Entry, ttk.Combobox)):
            return
        self.decidir(decision)

    def decidir(self, decision):
        if (
            not self.actual
            or self.lote_terminado
            or self.preparando
            or self.seleccionando_candidato
        ):
            return
        ambas_ausentes = decision == "referencia_incorrecta"
        if ambas_ausentes:
            foto_ref = any(
                r["anio"] == 2023 and r["imagen"] and Path(r["imagen"]).exists()
                for r in self._recursos_sitio()
            )
            if not foto_ref:
                messagebox.showinfo("2023 sin imagen", "No puedes afirmar ausencia en ambas fotos.")
                return
            decision = "ausente"
        anio = int(self.anio.get())
        disponibles = [
            r
            for r in self._recursos_sitio()
            if r["anio"] == anio and r["imagen"] and Path(r["imagen"]).exists()
        ]
        if not disponibles and decision in ("presente", "ausente"):
            messagebox.showinfo(
                "Imagen pendiente",
                "No hay imagen de esta campaña. Prepara el caso o conserva una duda / no evaluable.",
            )
            return
        self.guardar_borrador()
        referencia = None
        if ambas_ausentes:
            referencia = {**self.datos_referencia("referencia_incorrecta"), "corregir": True}
        elif (self.actual, 2023) not in self.observaciones:
            referencia = self.datos_referencia()
        self.almacen.guardar_observacion(
            self.actual,
            anio,
            decision,
            time.monotonic() - self.inicio,
            self.alcance.get(),
            [t.strip() for t in self.etiquetas.get().split(",") if t.strip()],
            self.notas.get(),
            referencia=referencia,
        )
        self.ultimo_guardado = f"Guardado: {self.actual} · {anio} · {NOMBRES[decision]}." + (
            f" También 2023: {NOMBRES[referencia['decision']]}." if referencia else ""
        )
        self.recargar()
        self.siguiente(tras_respuesta=True)

    def datos_referencia(self, decision=None):
        o = self.observaciones.get((self.actual, 2023), {})
        borrador = self.almacen.meta(f"borrador_{self.actual}_2023", {})
        return {
            "decision": decision
            or next(d for d, nombre in NOMBRES.items() if nombre == self.estado_referencia.get()),
            "alcance": borrador.get("alcance", o.get("alcance", self.alcance.get())),
            "etiquetas": [
                t.strip()
                for t in borrador.get(
                    "etiquetas", ", ".join(json.loads(o.get("etiquetas", "[]")))
                ).split(",")
                if t.strip()
            ],
            "notas": borrador.get("notas", o.get("notas", "")),
        }

    def corregir_referencia(self, decision=None):
        """Corrige la foto izquierda sin interrumpir los históricos del sitio."""
        if not self.actual:
            return
        datos = self.datos_referencia(decision)
        foto = any(
            r["anio"] == 2023 and r["imagen"] and Path(r["imagen"]).exists()
            for r in self._recursos_sitio()
        )
        if not foto and datos["decision"] in ("presente", "ausente", "referencia_incorrecta"):
            messagebox.showinfo(
                "2023 sin imagen", "La referencia solo permite duda o no evaluable."
            )
            self.dibujar()
            return
        o = self.observaciones.get((self.actual, 2023), {})
        self.almacen.guardar_observacion(
            self.actual,
            2023,
            datos["decision"],
            o.get("segundos", 0),
            datos["alcance"],
            datos["etiquetas"],
            datos["notas"],
        )
        self.ultimo_guardado = (
            f"Guardado: 2023 · {NOMBRES[datos['decision']]}. Continúa con {self.anio.get()}."
        )
        self.recargar()
        self.dibujar()
        self.raiz.focus_set()

    def abrir_cambios(self):
        self.menu_cambios.tk_popup(
            self.boton_cambios.winfo_rootx(),
            self.boton_cambios.winfo_rooty() + self.boton_cambios.winfo_height(),
        )

    def marcar_cambio_referencia(self, etiqueta):
        """Anota el cambio del par visible en 2023 sin repetir su clasificación."""
        if not self.actual or int(self.anio.get()) == 2023:
            return
        self.guardar_borrador()
        clave = f"borrador_{self.actual}_2023"
        o = self.observaciones.get((self.actual, 2023), {})
        borrador = self.almacen.meta(clave, {})
        etiquetas = [
            t.strip()
            for t in borrador.get(
                "etiquetas", ", ".join(json.loads(o.get("etiquetas", "[]")))
            ).split(",")
            if t.strip()
        ]
        if etiqueta not in etiquetas:
            etiquetas.append(etiqueta)
        notas = borrador.get("notas", o.get("notas", ""))
        nota = f"Comparación {self.anio.get()}→2023: {CAMBIOS[etiqueta]}."
        if nota not in notas.splitlines():
            notas = "\n".join(t for t in (notas, nota) if t)
        alcance = borrador.get("alcance", o.get("alcance", "emplazamiento_completo"))
        self.almacen.poner_meta(
            clave, {"etiquetas": ", ".join(etiquetas), "notas": notas, "alcance": alcance}
        )
        if o:
            self.almacen.guardar_observacion(
                self.actual, 2023, o["decision"], o["segundos"], alcance, etiquetas, notas
            )
        self.ultimo_guardado = (
            f"{'Guardado' if o else 'Borrador guardado'} en 2023: {CAMBIOS[etiqueta]}. "
            f"Continúa respondiendo a {self.anio.get()}; no necesitas repetir 2023."
        )
        self.recargar()
        self.dibujar()

    def guardar_campos(self, _=None):
        if self.actual:
            self.almacen.modificar_sitio(
                self.actual, soporte=self.soporte.get(), relacion=self.relacion.get()
            )
            self.guardar_borrador()
            self.recargar()
            self.dibujar()
            if _ is not None:
                self.raiz.focus_set()

    def borrador(self, _=None):
        self.guardar_borrador()

    def anadir_etiqueta(self, etiqueta):
        etiquetas = [t.strip() for t in self.etiquetas.get().split(",") if t.strip()]
        if etiqueta not in etiquetas:
            etiquetas.append(etiqueta)
        self.etiquetas.set(", ".join(etiquetas))
        self.guardar_borrador()

    def guardar_borrador(self, _=None):
        if self.actual:
            self.almacen.poner_meta(
                f"borrador_{self.actual}_{self.anio.get()}",
                {
                    "etiquetas": self.etiquetas.get(),
                    "notas": self.notas.get(),
                    "alcance": self.alcance.get(),
                },
            )
            o = self.observaciones.get((self.actual, int(self.anio.get())))
            if o and (
                o["notas"] != self.notas.get()
                or json.loads(o["etiquetas"])
                != [t.strip() for t in self.etiquetas.get().split(",") if t.strip()]
                or o["alcance"] != self.alcance.get()
            ):
                self.almacen.guardar_observacion(
                    self.actual,
                    int(self.anio.get()),
                    o["decision"],
                    o["segundos"],
                    self.alcance.get(),
                    [t.strip() for t in self.etiquetas.get().split(",") if t.strip()],
                    self.notas.get(),
                )
                self.observaciones[(self.actual, int(self.anio.get()))] = self.almacen.filas(
                    "SELECT * FROM observaciones WHERE emplazamiento=? AND anio=?",
                    (self.actual, int(self.anio.get())),
                )[0]

    def deshacer(self):
        accion = self.almacen.deshacer()
        self.recargar()
        if accion and accion["tipo"] in ("observacion", "comparacion", "sitio"):
            self.almacen.poner_meta(
                f"borrador_{accion['objetivo']}_{accion['anio'] or self.anio.get()}", {}
            )
            self.actual = None
            self.mostrar(accion["objetivo"], accion["anio"] or int(self.anio.get()))
        elif accion and accion["tipo"] == "agrupacion":
            anteriores = json.loads(accion["anterior"])
            self.cola = [s for s in self.cola if s not in anteriores["nuevos"]] + list(
                anteriores["activos"]
            )
            self.actual = None
            self.mostrar(next(iter(anteriores["activos"])), 2023)

    def confirmar_grupo(self):
        if self.actual:
            self.almacen.modificar_sitio(self.actual, agrupacion="confirmada")
            self.recargar()
            self.dibujar()

    def asociar(self):
        if not self.actual:
            return
        s = self.sitios[self.actual]
        texto = simpledialog.askstring(
            "Asociación independiente",
            "Identificadores internos de edificios, separados por comas; vacío para ninguna asociación:",
            initialvalue=", ".join(json.loads(s["edificios"])),
        )
        if texto is None:
            return
        cubiertas = simpledialog.askstring(
            "Cubiertas",
            "Identificadores de cubiertas, separados por comas. También puedes usar cubierta_manual_... para construcciones no catastradas:",
            initialvalue=", ".join(json.loads(s["cubiertas"])),
        )
        if cubiertas is None:
            return
        self.almacen.modificar_sitio(
            self.actual,
            edificios=[t.strip() for t in texto.split(",") if t.strip()],
            cubiertas=[t.strip() for t in cubiertas.split(",") if t.strip()],
            relacion=self.relacion.get(),
        )
        self.recargar()
        self.dibujar()

    def reagrupar(self):
        if not self.actual:
            return
        sitios = simpledialog.askstring(
            "Agrupación",
            "IDs de emplazamientos para dividir o fusionar, separados por comas:",
            initialvalue=self.actual,
        )
        if not sitios:
            return
        sitios = [s.strip() for s in sitios.split(",") if s.strip()]
        miembros = self.almacen.filas(
            "SELECT * FROM miembros WHERE emplazamiento IN (" + ",".join("?" for _ in sitios) + ")",
            tuple(sitios),
        )
        texto = simpledialog.askstring(
            "Repartir componentes",
            "Componentes separados por comas. Separa grupos nuevos con punto y coma (;). Para fusionar usa un solo grupo:",
            initialvalue=", ".join(m["componente"] for m in miembros),
        )
        if not texto:
            return
        grupos = [
            [c.strip() for c in linea.split(",") if c.strip()]
            for linea in texto.split(";")
            if linea.strip()
        ]
        try:
            nuevos = self.almacen.reorganizar(sitios, grupos)
        except ValueError as error:
            messagebox.showerror("Agrupación", str(error))
            return
        self.recargar()
        self.cola = [s for s in self.cola if s not in sitios] + nuevos
        self.mostrar(nuevos[0], 2023)

    def contexto(self):
        if not self.actual or self.preparando:
            return
        self.cancelar_candidato()
        sitio = self.actual

        def preparar(almacen, informar):
            from paneles_solares.etiquetado.preparar_seguimiento import ampliar_contexto

            ampliar_contexto(almacen, self.config, sitio, progreso=informar)

        self.iniciar_carga(preparar, "Ampliando contexto…", conservar_caso=True)

    def anadir(self):
        if not self.actual or self.preparando:
            return
        self.seleccionando_candidato = True
        self.dibujar()
        self.paso.config(
            text="Haz clic sobre los paneles omitidos en cualquiera de las fotos. Esc cancela."
        )
        self.raiz.focus_set()

    def cancelar_candidato(self):
        if self.seleccionando_candidato:
            self.seleccionando_candidato = False
            self.dibujar()

    def crear_candidato_en_foto(self, evento):
        """Reutiliza la foto pulsada para incorporar una instalación sin pedir coordenadas."""
        ancho, alto = max(evento.widget.winfo_width(), 100), max(evento.widget.winfo_height(), 100)
        lado = min(ancho, alto - 30)
        u = (evento.x - (ancho - lado) / 2) / lado
        v = (evento.y - (alto + 30 - lado) / 2) / lado
        if not (0 <= u <= 1 and 0 <= v <= 1):
            return
        xmin, ymin, xmax, ymax = self.extension_vista
        x = xmin + (self.centro[0] + (u - 0.5) / self.zoom) * (xmax - xmin)
        y = ymax - (self.centro[1] + (v - 0.5) / self.zoom) * (ymax - ymin)
        anio = 2023 if evento.widget == self.paneles[0] else int(self.anio.get())
        recursos = [
            r
            for r in self._recursos_sitio()
            if r["anio"] == anio
            and r["imagen"]
            and Path(r["imagen"]).exists()
            and r["xmin"] <= x <= r["xmax"]
            and r["ymin"] <= y <= r["ymax"]
        ]
        if not recursos:
            self.paso.config(text="Haz clic en una zona con imagen disponible. Esc cancela.")
            return
        existente = next(
            (
                o["id"]
                for o in self.objetivos
                if not o["provisional"] and o["geometria"].covers(Point(x, y))
            ),
            None,
        )
        if existente:
            self.mostrar(existente, int(self.anio.get()))
            self.raiz.focus_set()
            return
        try:
            from paneles_solares.etiquetado.preparar_seguimiento import anadir_candidato

            sid = anadir_candidato(self.almacen, x, y, ventana=recursos[0]["ventana"])
            self.recargar()
            if sid not in self.cola:
                self.cola.append(sid)
            self.mostrar(sid, int(self.anio.get()))
            self.raiz.focus_set()
        except ValueError as error:
            messagebox.showerror("Candidato", str(error), parent=self.raiz)

    def resumen(self):
        self.guardar_borrador()
        generar_resumen(self.almacen, geopackage=True)
        webbrowser.open((SALIDA / "resumen_correo.txt").as_uri())

    def cerrar(self):
        self.detener_anticipacion.set()
        if self.evento_anticipacion:
            self.raiz.after_cancel(self.evento_anticipacion)
            self.evento_anticipacion = None
        if self.evento_preparacion:
            self.raiz.after_cancel(self.evento_preparacion)
        self.guardar_borrador()
        generar_resumen(self.almacen)
        self.almacen.cerrar()
        self.raiz.destroy()
