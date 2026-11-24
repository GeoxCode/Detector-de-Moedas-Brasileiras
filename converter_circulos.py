import argparse
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = Path(__file__).resolve().parent
OUTPUT = ANNOTATIONS / "yolo_seg"
CLASS_NAMES = ["5", "10", "25", "50", "100"]
POINT_COUNT = 64


def clip_polygon(points: list[tuple[float, float]], width: int, height: int):
    boundaries = (
        (0, 0.0, True),
        (0, float(width - 1), False),
        (1, 0.0, True),
        (1, float(height - 1), False),
    )
    clipped = points
    for axis, boundary, keep_greater in boundaries:
        if not clipped:
            break
        result = []
        previous = clipped[-1]
        previous_inside = (
            previous[axis] >= boundary
            if keep_greater
            else previous[axis] <= boundary
        )
        for current in clipped:
            current_inside = (
                current[axis] >= boundary
                if keep_greater
                else current[axis] <= boundary
            )
            if current_inside != previous_inside:
                delta = current[axis] - previous[axis]
                if delta:
                    ratio = (boundary - previous[axis]) / delta
                    other_axis = 1 - axis
                    intersection = list(previous)
                    intersection[axis] = boundary
                    intersection[other_axis] = previous[other_axis] + ratio * (
                        current[other_axis] - previous[other_axis]
                    )
                    result.append(tuple(intersection))
            if current_inside:
                result.append(current)
            previous = current
            previous_inside = current_inside
        clipped = result
    return clipped


def convert_split(split: str) -> tuple[int, int]:
    annotation_dir = ANNOTATIONS / f"{split}_json"
    image_dir = ROOT / "datast_Novo_cinza" / split / "images"
    output_split = OUTPUT / split
    output_images = output_split / "images"
    output_labels = output_split / "labels"

    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Pasta de anotações não encontrada: {annotation_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {image_dir}")
    if output_split.exists() and any(output_split.iterdir()):
        raise FileExistsError(
            f"A saída de {split} já contém arquivos; não vou sobrescrever: "
            f"{output_split}"
        )

    prepared = []
    clipped_count = 0
    json_files = sorted(annotation_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"Nenhum JSON encontrado em {annotation_dir}")

    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        image_path = image_dir / Path(data["imagePath"]).name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Referência de imagem inválida em {json_path.name}: {image_path}"
            )

        rows = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label not in CLASS_NAMES:
                raise ValueError(f"Classe inválida em {json_path.name}: {label!r}")
            if shape.get("shape_type") != "circle":
                raise ValueError(
                    f"Esperado círculo em {json_path.name}; recebido "
                    f"{shape.get('shape_type')!r}"
                )
            points = shape.get("points", [])
            if len(points) != 2:
                raise ValueError(f"Círculo inválido em {json_path.name}")

            (cx, cy), (px, py) = points
            radius = math.hypot(px - cx, py - cy)
            if radius <= 0:
                raise ValueError(f"Círculo sem área em {json_path.name}")
            circle = [
                (
                    cx + radius * math.cos(2 * math.pi * index / POINT_COUNT),
                    cy + radius * math.sin(2 * math.pi * index / POINT_COUNT),
                )
                for index in range(POINT_COUNT)
            ]
            polygon = clip_polygon(circle, width, height)
            if len(polygon) < 3:
                raise ValueError(f"Polígono sem área suficiente em {json_path.name}")
            if len(polygon) != POINT_COUNT:
                clipped_count += 1

            class_id = CLASS_NAMES.index(label)
            coordinates = " ".join(
                f"{max(0.0, min(1.0, x / width)):.8f} "
                f"{max(0.0, min(1.0, y / height)):.8f}"
                for x, y in polygon
            )
            rows.append(f"{class_id} {coordinates}")

        if not rows:
            raise ValueError(f"Nenhum objeto anotado em {json_path.name}")
        prepared.append((image_path, json_path.stem, "\n".join(rows) + "\n"))

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    for image_path, stem, content in prepared:
        shutil.copy2(image_path, output_images / image_path.name)
        (output_labels / f"{stem}.txt").write_text(content, encoding="utf-8")

    print(
        f"{split}: {len(prepared)} imagens convertidas; "
        f"{clipped_count} círculos recortados nos limites da imagem."
    )
    return len(prepared), clipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte círculos do Labelme em polígonos YOLO Segmentation."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=["train"],
        help="Divisões com anotações Labelme prontas (padrão: train).",
    )
    args = parser.parse_args()

    total = sum(convert_split(split)[0] for split in args.splits)
    print(f"Concluído: {total} imagens preparadas em {OUTPUT}")
    print(f"Ordem das classes: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
