# Reto hackathon: detección y caracterización de un incendio con Sentinel-2

## Objetivo
Construir un pipeline en Python que, a partir de **dos imágenes Sentinel-2 de la misma escena
(True Color y SWIR)**, detecte y caracterice un incendio forestal: zona quemada, frente activo
y humo, con métricas (hectáreas, perímetro, centroide, dirección del humo y de propagación) y
salidas visuales (overlays, GeoJSON, mapa web).

Estamos practicando con el incendio del **26-07-2026** al SO de la Comunidad de Madrid
(límite con Ávila y Toledo; Valle de Iruelas y carretera M-501). El reto real será parecido.

## IMPORTANTE: dos modos de uso
En el reto real solo habrá **dos imágenes (True Color y SWIR)**, posiblemente PNG/JPG sin
georreferencia. No tendremos bandas crudas ni escena previa. Por eso:

- **Modo reto:** el pipeline debe funcionar solo con TC + SWIR. La escala sale de la barra
  de escala de la imagen o de un parámetro `--px-per-km`. Sin dNBR, sin bandas.
- **Modo calibración:** con los GeoTIFF de práctica (bandas + escena previa) se calcula el
  dNBR y se genera una máscara de quemado de **referencia**. Con ella se mide la
  segmentación por color del modo reto (IoU, precisión, recall) y se fijan los umbrales.
  El dNBR es herramienta de validación, no parte de la solución final.

## Las imágenes del reto serán distintas a las de práctica
Todo lo que tenemos (GeoTIFF y capturas) lo descargó Dani a mano de Copernicus Browser para
practicar. **No son las imágenes del hackathon.** Las del reto pueden tener otra combinación
de bandas en el SWIR (colores distintos), otro procesado del True Color, otra resolución,
otro formato y otra escala. Consecuencias para el diseño:
- No fijar colores absolutos en el código. Los umbrales deben ser **adaptativos** (por
  percentiles, Otsu, k-means o similar sobre la propia imagen) y vivir en un fichero de
  configuración.
- Incluir un paso de **calibración rápida** al recibir las imágenes reales: ver una vista
  previa, marcar unas muestras de píxeles (quemado, humo, fuego, fondo) y recalcular los
  umbrales a partir de ellas, en pocos minutos.
- Validar la robustez con las imágenes de práctica en varias versiones (8 bits, JPG/PNG,
  menor resolución, otra combinación de bandas SWIR generada desde las bandas crudas).
- Detectar y avisar si la escala no se puede leer, en vez de asumir un valor.

## Datos disponibles
Todo en EPSG:32630 (UTM 30N). No mezclar con otros lotes de descarga.

- `datos_16bit/` (18 GeoTIFF, mismas fechas y misma rejilla):
  - Fechas: `2026-06-24` (previa al incendio, casi sin nubes) y `2026-07-26` (incendio).
  - Bandas crudas: B02, B03, B04, B8A, B11, B12 en uint16. Reflectancia = valor / 65535.
  - Capas visuales: True_color, SWIR, Wildfires en uint16 escalado a 0-65535
    (equivale a 8 bits x 257). Dividir por 257 para volver a 0-255 y reutilizar umbrales.
  - Rejilla: 2500 x 1128 px, píxel **34,95 x 33,45 m (no cuadrado)**, esquina superior
    izquierda (330246, 4490363). Extensión aprox. 87 x 38 km.
- `vista_amplia/`: True color y SWIR del 26-07, 289 x 280 m/px, 2500 x 1128 px. Tiene zona
  negra sin datos (0,0,0) en el borde de la pasada. Sirve para medir el humo completo
  (~150-200 km de largo), que no cabe en la vista de detalle.
- `prototipo/`: `incendio_pipeline.py` (prototipo hecho sobre capturas de pantalla de
  Copernicus Browser que sacó Dani a mano, sin georreferencia, escala leída de la barra de
  10 km) y esas capturas. Son solo material de práctica y NO son las imágenes del reto.
  Úsalo como base de ideas, no como código final.

## Qué hay en la escena
- Cicatriz grande en el centro, con zonas sin quemar dentro (islas verdes) y núcleos urbanos.
- Embalses que salen casi negros en SWIR (no confundir con quemado oscuro).
- Humo hacia el SSE, que cubre parte de la cicatriz sur el 26-07 (el SWIR lo atraviesa en
  parte; en SWIR se ve como neblina azulada).
- Fuego activo visible en Wildfires: un frente en el lado este, trazos en el borde sur bajo el
  humo y focos sueltos en el centro-sur.
- Costuras entre teselas Sentinel-2 en True Color: no son rasgos reales.
- Referencia aproximada (comprobación rápida, 8 y 16 bits dan lo mismo): fuera de la cicatriz
  el dNBR tiene mediana ~+0,02 y desviación ~0,04. Área con dNBR > 0,1 / 0,27 / 0,44 / 0,66:
  ~620 / ~348 / ~204 / ~81 km2. Úsalo solo para comprobar que el resultado es coherente.

## Plan por fases (PARAR al final de cada una)
Al terminar cada fase: guardar figuras y tablas en `resultados/faseN/`, mostrar un resumen con
cifras y rutas, y esperar confirmación. No avanzar sin ella.

### Fase 0 - Preparación
- Entorno y `requirements.txt` (rasterio, numpy, scipy, scikit-image, opencv, matplotlib,
  geopandas, shapely, folium).
- Leer los GeoTIFF, comprobar que las 18 comparten rejilla, CRS y escalas, y confirmar la
  conversión a reflectancia. Informe breve de datos.

### Fase 1 - dNBR y máscara de referencia
- NBR = (B8A - B12) / (B8A + B12) en ambas fechas; dNBR = NBR_previa - NBR_incendio.
- Severidad con umbrales USGS (0,1 / 0,27 / 0,44 / 0,66).
- Enmascarar agua (NDWI o B8A bajo), nubes y sombras (junio casi limpio) y sin datos.
- No rellenar huecos de la máscara: respetar las islas sin quemar y los núcleos urbanos.
- Salidas: mapa de dNBR, mapa de severidad, tabla de áreas (ha y km2) usando el tamaño real
  de píxel (ancho != alto), máscara de referencia binaria.

### Fase 2 - Segmentación solo con TC + SWIR (modo reto)
- Quemado en SWIR con dos reglas: rojizo/marrón (tono bajo, saturado, R > G) y
  oscuro/oliva (G y B bajos, R ~ G). Con solo la primera se pierde mucha superficie.
- Excluir agua. Limpieza morfológica sin rellenar huecos.
- Medir contra la referencia de la Fase 1: IoU, precisión y recall. Ajustar umbrales con esos
  datos, no a ojo.
- Probar robustez: pasar las capas a 8 bits, a JPG/PNG y reducir resolución, y repetir.

### Fase 3 - Frente activo, humo y métricas
- Frente activo: píxeles naranja/amarillo muy brillantes en Wildfires y en SWIR (en
  calibración, apoyar con B11/B12 altos). Si en modo reto no hay píxeles saturados, inferirlo
  como borde de la cicatriz pegado al humo y etiquetarlo "inferido".
- Humo: en True Color es gris/azulado y claro (R - B pequeño, baja saturación), mientras que
  el suelo es cálido (R > B). Quedarse solo con manchas pegadas a la cicatriz, para
  descartar ciudades. Extensión total desde `vista_amplia/`.
- Penacho: PCA para largo, ancho y rumbo (origen -> extremo). El viento probable se deduce
  solo de ahí: indicarlo como orientativo (contrastable con ERA5/AEMET).
- Propagación: vector desde el centroide de la cicatriz al frente. Con una sola fecha de
  incendio es orientativo; dejarlo dicho.
- Métricas: hectáreas por clase y severidad, perímetro (contorno simplificado; avisar de que
  es aproximado), centroide, caja envolvente, longitud del frente, rumbo del penacho.

### Fase 4 - Salidas y empaquetado
- Overlays sobre True Color y SWIR, y un zoom con flechas, barra de escala y norte.
- `metricas.json`, máscaras PNG y GeoJSON en EPSG:4326. Sin georreferencia: coordenadas de
  píxel, con opción `--bounds W S E N`.
- Mapa web con Folium y capas activables.
- Un único punto de entrada: `python run.py tc.png swir.png [--px-per-km X] [--bounds ...]`.
- Probarlo con las capturas de `prototipo/` (sin georreferencia) y con los GeoTIFF.
- README de una página y `requirements.txt`; que corra en un entorno limpio (p. ej. Colab).

## Trampas conocidas
- Quemado oscuro/oliva: ver Fase 2. Embalses y sombras pueden confundirse con él.
- Ciudades y suelo desnudo pueden parecerse al humo en True Color.
- Tejados rojos y suelo desnudo pueden dar falsos positivos de quemado.
- El píxel no es cuadrado: usar ambos valores en áreas y distancias.
- El humo se corta en los bordes de la imagen: su largo es un mínimo salvo en la vista amplia.
- Nubes y sombras de nubes en la escena previa, aunque junio 24 está casi limpia.

## Validación opcional
- NASA FIRMS (CSV de puntos de calor, 25-27 de julio) para comprobar el frente.
- Perímetros EFFIS / Copernicus EMS si están disponibles.

## Reglas de trabajo
- Todo número sale de las máscaras y la georreferencia. No inventar cifras. Indicar
  incertidumbres y qué es estimación.
- Parámetros y umbrales en un único fichero de configuración, no repartidos por el código.
- Código claro, comentarios en español, funciones pequeñas y reutilizables.
- No modificar los datos originales. Salidas en `resultados/`.
- Si algo es ambiguo, preguntar antes de decidir.