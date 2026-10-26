/* El HTML contiene datos y agregados. Cambiar filtros no modifica los originales. */
const datos = JSON.parse(document.getElementById('datos').textContent);
const elemento = id => document.getElementById(id);
const numero = (valor, decimales = 0) => valor == null ? 'Sin dato' :
    Number(valor).toLocaleString('es-ES', { maximumFractionDigits: decimales });
const texto = valor => String(valor ?? '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const etiquetas = {
    comprobar_edificio: 'Comprobar edificio', resolver_atribucion: 'Resolver atribución',
    comprobar_emplazamiento_exterior: 'Comprobar emplazamiento exterior',
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

function actualizarGraficos() {
    const f = filtros();
    const cohorte = candidatosFiltrados('muestra');
    const campo = f.metrica === 'con_mascara' ? 'presentes' : f.metrica;
    const titulos = { superficie_m2: 'Superficie observada', presentes: 'Candidatos con presencia', produccion_central_mwh: 'Producción estimada · MWh/año' };
    elemento('evolucion-titulo').textContent = titulos[campo];
    barras('evolucion', datos.comparables.map(anio => [anio, resumen(anio, f.municipio, 'muestra')?.[campo]]), campo === 'superficie_m2' ? ' m²' : '');
    elemento('cohorte').textContent = datos.comparables.length ?
        `${numero(cohorte.length)} candidatos con área comparable en ${datos.comparables.join(', ')}.` :
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
    elemento('cambios-nota').textContent = `${numero(conPaneles.length)} candidatos con paneles en alguna de las dos fechas. ` +
        `${numero(parejas.length - conPaneles.length)} ausentes en ambas fechas, separados del histograma. ` +
        `Aparecen: ${numero(aparecen)}; desaparecen: ${numero(desaparecen)}. ` +
        'Área horizontal anotada: un cambio pequeño puede proceder del dibujo o de la imagen; no prueba una ampliación.';
}

function actualizarTotales() {
    const f = filtros();
    const r = resumen(f.anio, f.municipio);
    const tarjetas = [
        [numero(r?.superficie_m2, 1), 'm² horizontales observados', fuenteCampania(f.anio)],
        [numero(r?.con_mascara ?? 0), 'candidatos con máscara', 'Agrupaciones provisionales, no instalaciones confirmadas'],
        [numero(r?.ausentes ?? 0), 'ausencias confirmadas', 'Solo los candidatos y la campaña de la selección'],
        [numero(r?.sin_estado_resuelto ?? 0), 'estados pendientes', 'Incluye detecciones sin confirmar y vínculos pendientes'],
    ];
    elemento('totales').innerHTML = tarjetas.map(([valor, titulo, nota]) =>
        `<div class="total"><strong>${valor}</strong><span>${titulo}</span><small>${nota}</small></div>`).join('');
    elemento('alcance').textContent = modo === 'muestra' ?
        `Muestra común de ${datos.comparables.join(', ')}: mismos candidatos y superficies revisadas en todas las fechas.` :
        'Inventario seleccionado, no censo regional. La cobertura revisada cambia entre campañas; sus totales no miden por sí solos el crecimiento.';
}

function actualizarEnergia() {
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
    const tipo = c.instalacion_confirmada ? 'Instalación confirmada' : 'Candidato';
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
    mapa.setView([c.lat, c.lon], Math.max(19, mapa.getZoom()));
    dibujarDetalle();
}

function actualizar() {
    if (seleccion && !candidatosFiltrados().some(c => c.id_instalacion === seleccion)) seleccion = null;
    actualizarTotales();
    actualizarEnergia();
    actualizarGraficos();
    verFicha();
    const f = filtros();
    const maximo = maximoEscala();
    limiteEscala = maximo;
    elemento('leyenda-titulo').textContent = {
        superficie_m2: 'Superficie horizontal observada · m²', con_mascara: 'Candidatos con máscara',
        produccion_central_mwh: 'Producción estimada del inventario · MWh/año',
    }[f.metrica];
    elemento('escala').innerHTML = [0, .25, 1].map(t => `<span>${numero(maximo * t)}</span>`).join('');
    if (!mapa) return;
    municipios.setStyle(estiloMunicipio);
    ortofoto.setUrl(`https://geoservicios.cantabria.es/inspire/rest/services/Ortofoto_${f.anio}_WebMercator/MapServer/tile/{z}/{y}/{x}`);
    dibujarDetalle();
}

function crearMapa() {
    if (!window.L) {
        elemento('mapa').innerHTML = '<p class="aviso">No se pudo cargar el mapa. Comprueba la conexión. Los datos y gráficos siguen disponibles.</p>';
        return;
    }
    mapa = L.map('mapa', { preferCanvas: true, maxZoom: 22 }).setView([43.20, -4.04], 9);
    const calles = L.tileLayer('https://tms-ign-base.idee.es/1.0.0/IGNBaseTodo/{z}/{x}/{-y}.jpeg', {
        maxNativeZoom: 19, maxZoom: 22,
        attribution: '<a href="https://www.ign.es">© Instituto Geográfico Nacional</a>' }).addTo(mapa);
    ortofoto = L.tileLayer('', { maxNativeZoom: 19, maxZoom: 22,
        attribution: 'Ortofoto de la campaña · Gobierno de Cantabria / IGN-CNIG' });
    municipios = L.geoJSON(datos.municipios, {
        style: estiloMunicipio,
        onEachFeature: (feature, capa) => {
            capa.bindTooltip(() => {
                const r = resumen(filtros().anio, feature.properties.codigo);
                return `<strong>${texto(feature.properties.municipio)}</strong><br>` +
                    `${numero(r?.superficie_m2, 1)} m² observados<br>${numero(r?.con_mascara ?? 0)} candidatos con máscara`;
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
    L.control.layers({ 'Mapa de calles · IGN': calles, 'Ortofoto de la campaña': ortofoto },
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
crearMapa();
mostrarSupuestosEnergia();
actualizar();
