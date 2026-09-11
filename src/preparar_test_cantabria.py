import json
import shutil
from pathlib import Path

CARPETA_IMGS = Path("./data/labeling/test/images")
CARPETA_JSON = Path("./data/labeling/test/annotations")

RUTA_YOLO = Path("./data/processed/cantabria_test_yolo")
RUTA_YAML = RUTA_YOLO / "cantabria_test.yaml"
RUTA_IMG_YOLO = RUTA_YOLO / "images"
RUTA_LAB_YOLO = RUTA_YOLO / "labels"

TAM_IMG = 512


def convierte_json_yolo(ruta_json_ori: Path, ruta_txt_dst: Path) -> int:
    """Funcion que convierte los anotations hechos por labelme a algo que
    ultralytics entiende (txt con px)

    Args:
        ruta_json_ori (Path): Json origen
        ruta_txt_dst (Path): Txt destino

    Returns:
        int: Retorna el num de paneles que habia
    """
    # Leemos los contenidos del json
    contenido_json = json.loads(ruta_json_ori.read_text(encoding="utf-8"))
    total = len(contenido_json["shapes"])
    lineas = []  # Una por cada panel
    for panel in contenido_json["shapes"]:  # Puede haber varios por img
        px_panel = panel["points"]  # Sacamos la lista de puntos

        pixeles = []
        # Viene como una lista de puntos en R^2
        for x, y in px_panel:
            # Yolo necesita (0, 1) no px absolutos
            pixeles.append(x / TAM_IMG)
            pixeles.append(y / TAM_IMG)

        txt = " ".join(f"{valor:.6f}" for valor in pixeles)

        lineas.append(f"0 {txt}")  # 0 es la clase de panel_solar

    ruta_txt_dst.parent.mkdir(parents=True, exist_ok=True)
    ruta_txt_dst.write_text("\n".join(lineas), encoding="utf-8")
    return total


def preparar_yaml() -> None:
    """Crea la configuración para Ultralytics"""

    contenido = (
        f"path: {RUTA_YOLO.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "test: images\n"
        "\n"
        "names:\n"
        "  0: panel_solar\n"
    )

    RUTA_YAML.parent.mkdir(parents=True, exist_ok=True)
    RUTA_YAML.write_text(data=contenido, encoding="utf-8")


def preparar_test():
    """Funcion que prepara todos los datos del label me
    para que valgan para los test de ultralytics

    Raises:
        ValueError: En caso de que la imagen correspondiente a un json
        no exista
    """
    # Sacamos todos los json
    jsons = sorted(CARPETA_JSON.glob(pattern="*.json"))

    # Creamos por si no existe
    RUTA_YOLO.mkdir(parents=True, exist_ok=True)
    RUTA_IMG_YOLO.mkdir(parents=True, exist_ok=True)
    RUTA_LAB_YOLO.mkdir(parents=True, exist_ok=True)

    total = 0  # Por llevar un recuento

    for idx, json_lab in enumerate(iterable=jsons, start=1):
        print(f"Procesando json [{idx}/{len(jsons)}]")
        # Sacamos solo el nombre
        name = json_lab.stem

        # Como el nombre ha de ser el mismo
        rut_img_ori = CARPETA_IMGS / f"{name}.png"
        rut_img_dst = RUTA_IMG_YOLO / f"{name}.png"
        rut_lab_dst = RUTA_LAB_YOLO / f"{name}.txt"

        # Copiamos la imagen a la carpeta de los tests
        if rut_img_ori.exists():
            shutil.copy2(src=rut_img_ori, dst=rut_img_dst)
        else:
            print("Imagen no encontrada")
            raise ValueError

        total += convierte_json_yolo(ruta_json_ori=json_lab, ruta_txt_dst=rut_lab_dst)

    print(f"Se han encontrado {total} paneles")

    # Necesitamos el YAML para correr los tests
    preparar_yaml()


def main():
    """Programa que prepara todos los datos de labelme manuales para test yolo"""
    preparar_test()


if __name__ == "__main__":
    main()
