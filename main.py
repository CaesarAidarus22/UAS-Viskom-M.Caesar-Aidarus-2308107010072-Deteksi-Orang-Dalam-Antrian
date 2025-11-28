import cv2
import argparse
from detect import load_yolo, detect_people
from preprocess import preprocess_image
import os

def classify_density(count):
    if count <= 2:
        return "SEPI", (0,255,0)        # Hijau
    elif count <= 5:
        return "SEDANG", (0,255,255)    # Kuning
    else:
        return "RAMAI", (0,0,255)       # Merah

def draw_results(img, boxes, density_label, color):
    # Gambar bounding box
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Tampilkan jumlah orang
    cv2.putText(img, f"Jumlah Orang: {len(boxes)}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Tampilkan status kepadatan
    cv2.putText(img, f"Status: {density_label}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img

def main(image_path):
    # Load gambar input
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Gambar tidak ditemukan!")
        return

    print("[INFO] Preprocessing image...")
    gray, blur, edges = preprocess_image(img)   # Tidak wajib dipakai di hasil akhir

    print("[INFO] Loading YOLO model...")
    net, output_layers = load_yolo()

    print("[INFO] Detecting people...")
    boxes = detect_people(img, net, output_layers)

    density, color = classify_density(len(boxes))
    print(f"[INFO] Detected {len(boxes)} people → Status: {density}")

    result_img = draw_results(img, boxes, density, color)

    # Simpan output
    os.makedirs("output", exist_ok=True)
    output_path = "output/hasil.jpg"
    cv2.imwrite(output_path, result_img)

    print(f"[INFO] Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    main(args.image)