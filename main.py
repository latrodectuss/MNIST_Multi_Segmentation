import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame
from PIL import Image, ImageTk, ImageDraw
import gdown
import os

class DigitSegmentationNet(nn.Module):
    def __init__(self, num_classes=11):
        super(DigitSegmentationNet, self).__init__()

        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.dec4 = self.up_conv_block(512, 256)
        self.dec3 = self.up_conv_block(512, 128)
        self.dec2 = self.up_conv_block(256, 64)
        self.dec1 = self.up_conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], 1))

        out = self.final_conv(d1)

        return out


class DigitSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Segmentation Demo")
        self.root.geometry("800x600")

        self.colors = [
            (255, 0, 0),  # 0: красный
            (0, 255, 0),  # 1: зеленый
            (0, 0, 255),  # 2: синий
            (255, 255, 0),  # 3: желтый
            (255, 0, 255),  # 4: пурпурный
            (0, 255, 255),  # 5: голубой
            (255, 128, 0),  # 6: оранжевый
            (128, 0, 255),  # 7: фиолетовый
            (0, 255, 128),  # 8: весенне-зеленый
            (128, 128, 255),  # 9: светло-синий
            (255, 255, 255)  # фон: белый
        ]

        self.model = self.load_model()

        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.brush_size = 15

        self.create_widgets()



    def load_model(self):
        model = DigitSegmentationNet(num_classes=11)

        file_id = "14_JJHPB23u-h2MBZ6BuXZFkxRj56sS30"
        url = f'https://drive.google.com/uc?id={file_id}'
        output_path = 'model_weights.pth'

        if not os.path.exists(output_path):
            print("Downloading model weights...")
            gdown.download(url, output_path, quiet=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(output_path, map_location=device))
        model.eval()
        model.to(device)

        print("Модель загружена")
        return model

    def create_widgets(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        instruction_label = Label(left_frame,
                                  text="Нарисуйте мышкой 1-4 цифры",
                                  font=('Arial', 14, 'bold'),
                                  fg='darkblue')
        instruction_label.pack(pady=10)

        self.canvas = Canvas(left_frame, width=256, height=256, bg='black', cursor="cross")
        self.canvas.pack(pady=10)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        button_frame = Frame(left_frame)
        button_frame.pack(pady=10)

        Button(button_frame, text="Сегментировать", command=self.segment_image,
               bg='green', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        Button(button_frame, text="Очистить", command=self.clear_canvas,
               bg='red', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        center_frame = Frame(main_frame)
        center_frame.pack(side=tk.LEFT, padx=20)

        input_frame = Frame(center_frame)
        input_frame.pack(pady=10)

        Label(input_frame, text="Входящее изображение", font=('Arial', 12)).pack()
        self.input_label = Label(input_frame)
        self.input_label.pack(pady=5)

        result_frame = Frame(center_frame)
        result_frame.pack(pady=10)

        Label(result_frame, text="Результат сегментации", font=('Arial', 12)).pack()
        self.result_label = Label(result_frame)
        self.result_label.pack(pady=5)

        self.digits_label = Label(center_frame, text="Найденные цифры: ",
                                  font=('Arial', 14), fg='blue')
        self.digits_label.pack(pady=10)

        right_frame = Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10)

        Label(right_frame, text="Легенда", font=('Arial', 12, 'bold')).pack(pady=10)

        for digit in range(10):
            color_frame = Frame(right_frame)
            color_frame.pack(anchor='w', pady=3)

            color_canvas = Canvas(color_frame, width=25, height=25, bg=self.get_color_hex(digit))
            color_canvas.pack(side=tk.LEFT, padx=5)

            Label(color_frame, text=f"Цифра {digit}", font=('Arial', 11)).pack(side=tk.LEFT)

    def get_color_hex(self, digit):
        r, g, b = self.colors[digit]
        return f'#{r:02x}{g:02x}{b:02x}'

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=self.brush_size, fill='white',
                                    capstyle=tk.ROUND, smooth=tk.TRUE)
            self.last_x = x
            self.last_y = y

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.input_label.config(image='')
        self.result_label.config(image='')
        self.digits_label.config(text="Найденные цифры: ")

    def get_canvas_image(self):
        pil_image = Image.new('L', (256, 256), color=0)
        draw = ImageDraw.Draw(pil_image)

        items = self.canvas.find_all()

        for item in items:
            coords = self.canvas.coords(item)
            color = self.canvas.itemcget(item, "fill")
            width = float(self.canvas.itemcget(item, "width"))

            if color.lower() == 'white':
                for i in range(0, len(coords) - 2, 2):
                    x1, y1, x2, y2 = coords[i], coords[i + 1], coords[i + 2], coords[i + 3]
                    draw.line([(x1, y1), (x2, y2)], fill=255, width=int(width))

        return pil_image

    def segment_image(self):
        try:
            pil_image = self.get_canvas_image()

            input_display = pil_image.resize((128, 128), Image.Resampling.LANCZOS)
            input_photo = ImageTk.PhotoImage(input_display)
            self.input_label.config(image=input_photo)
            self.input_label.image = input_photo

            input_tensor = self.preprocess_image(pil_image)

            with torch.no_grad():
                device = next(self.model.parameters()).device
                input_tensor = input_tensor.to(device)
                output = self.model(input_tensor)

                pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

            found_digits = self.find_digits(pred_mask)

            colored_mask = self.create_colored_mask(pred_mask)

            result_pil = Image.fromarray(colored_mask)
            result_display = result_pil.resize((128, 128), Image.Resampling.NEAREST)
            result_photo = ImageTk.PhotoImage(result_display)
            self.result_label.config(image=result_photo)
            self.result_label.image = result_photo

            digits_text = f"Найденные цифры: {', '.join(map(str, found_digits))}" if found_digits else "Цифры не найдены"
            self.digits_label.config(text=digits_text)

            print("Сегментация выполнена!")
            print(f"Найденные цифры: {found_digits}")

        except Exception as e:
            print(f"Ошибка: {e}")

    def find_digits(self, pred_mask):
        found_digits = []
        for digit in range(10):
            if np.any(pred_mask == digit):
                area = np.sum(pred_mask == digit)
                if area > 10:
                    found_digits.append(digit)

        return sorted(found_digits)

    def preprocess_image(self, pil_image):
        img_resized = pil_image.resize((64, 64), Image.Resampling.LANCZOS)

        img_array = np.array(img_resized)

        img_array = img_array.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

        return img_tensor

    def create_colored_mask(self, pred_mask):
        h, w = pred_mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(11):
            mask = pred_mask == class_id
            colored_mask[mask] = self.colors[class_id]

        return colored_mask

    def create_mask_display(self, pred_mask):
        mask_display = (pred_mask * 23).astype(np.uint8)  # 0-10 -> 0-230
        mask_pil = Image.fromarray(mask_display, mode='L')
        return mask_pil.resize((128, 128), Image.Resampling.NEAREST)


def main():
    root = tk.Tk()

    app = DigitSegmentationApp(root)

    root.mainloop()


if __name__ == "__main__":
    main()