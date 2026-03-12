"""
Graphical user interface for the pneumonia detection system.

This module provides a Tkinter-based interface that interacts with the
prediction pipeline through the integrator module.
"""

from tkinter import END, StringVar, Text, Tk, filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

import csv
from pathlib import Path

import img2pdf
import tkcap
from PIL import Image, ImageTk

from src.integrator import run_prediction_pipeline


MODEL_PATH = "models/conv_MLP_84.h5"


class App:
    """
    Graphical application for pneumonia classification.
    """

    def __init__(self) -> None:
        """
        Initialize the graphical interface.
        """
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        bold_font = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        self.label_title_left = ttk.Label(
            self.root,
            text="Imagen Radiográfica",
            font=bold_font,
        )
        self.label_title_right = ttk.Label(
            self.root,
            text="Imagen con Heatmap",
            font=bold_font,
        )
        self.label_result = ttk.Label(
            self.root,
            text="Resultado:",
            font=bold_font,
        )
        self.label_id = ttk.Label(
            self.root,
            text="Cédula Paciente:",
            font=bold_font,
        )
        self.label_header = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold_font,
        )
        self.label_probability = ttk.Label(
            self.root,
            text="Probabilidad:",
            font=bold_font,
        )

        self.patient_id = StringVar()

        self.entry_id = ttk.Entry(self.root, textvariable=self.patient_id, width=10)

        self.image_box_original = Text(self.root, width=31, height=15)
        self.image_box_heatmap = Text(self.root, width=31, height=15)
        self.text_result = Text(self.root)
        self.text_probability = Text(self.root)

        self.button_predict = ttk.Button(
            self.root,
            text="Predecir",
            state="disabled",
            command=self.run_model,
        )
        self.button_load = ttk.Button(
            self.root,
            text="Cargar Imagen",
            command=self.load_img_file,
        )
        self.button_clear = ttk.Button(
            self.root,
            text="Borrar",
            command=self.clear_interface,
        )
        self.button_save = ttk.Button(
            self.root,
            text="Guardar",
            command=self.save_results_csv,
        )
        self.button_pdf = ttk.Button(
            self.root,
            text="PDF",
            command=self.create_pdf,
        )

        self.label_title_left.place(x=110, y=65)
        self.label_title_right.place(x=545, y=65)
        self.label_result.place(x=500, y=350)
        self.label_id.place(x=65, y=350)
        self.label_header.place(x=122, y=25)
        self.label_probability.place(x=500, y=400)

        self.button_predict.place(x=220, y=460)
        self.button_load.place(x=70, y=460)
        self.button_clear.place(x=670, y=460)
        self.button_save.place(x=370, y=460)
        self.button_pdf.place(x=520, y=460)

        self.entry_id.place(x=200, y=350)
        self.text_result.place(x=610, y=350, width=90, height=30)
        self.text_probability.place(x=610, y=400, width=90, height=30)
        self.image_box_original.place(x=65, y=90)
        self.image_box_heatmap.place(x=500, y=90)

        self.entry_id.focus_set()

        self.image_path = None
        self.label = None
        self.probability = None
        self.original_preview = None
        self.heatmap_preview = None
        self.report_id = 0

        self.root.mainloop()

    def load_img_file(self) -> None:
        """
        Open a file dialog and load an image path selected by the user.
        """
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("Archivos de imagen", "*.dcm *.jpeg *.jpg *.png"),
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*"),
            ),
        )

        if not filepath:
            return

        self.image_path = filepath

        try:
            preview_image = Image.open(filepath)
        except Exception:
            showinfo(
                title="Carga de imagen",
                message=(
                    "La previsualización directa no está disponible para este "
                    "archivo, pero la imagen sí puede ser procesada."
                ),
            )
            self.button_predict["state"] = "enabled"
            return

        preview_image = preview_image.resize((250, 250), Image.Resampling.LANCZOS)
        self.original_preview = ImageTk.PhotoImage(preview_image)

        self.image_box_original.delete("1.0", END)
        self.image_box_original.image_create(END, image=self.original_preview)

        self.button_predict["state"] = "enabled"

    def run_model(self) -> None:
        """
        Execute the prediction pipeline and display the results.
        """
        if self.image_path is None:
            showinfo(title="Error", message="Primero cargue una imagen.")
            return

        result = run_prediction_pipeline(self.image_path, MODEL_PATH)

        self.label = result["label"]
        self.probability = result["probability"]

        heatmap_image = Image.fromarray(result["heatmap"])
        heatmap_image = heatmap_image.resize((250, 250), Image.Resampling.LANCZOS)
        self.heatmap_preview = ImageTk.PhotoImage(heatmap_image)

        self.image_box_heatmap.delete("1.0", END)
        self.image_box_heatmap.image_create(END, image=self.heatmap_preview)

        self.text_result.delete("1.0", END)
        self.text_probability.delete("1.0", END)

        self.text_result.insert(END, self.label)
        self.text_probability.insert(END, f"{self.probability:.2f}%")

    def save_results_csv(self) -> None:
        """
        Save the prediction results to a CSV file.
        """
        if self.label is None or self.probability is None:
            showinfo(title="Guardar", message="Primero ejecute una predicción.")
            return

        with open("historial.csv", "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    self.entry_id.get(),
                    self.label,
                    f"{self.probability:.2f}%",
                    Path(self.image_path).name if self.image_path else "",
                ]
            )

        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self) -> None:
        """
        Create a PDF report from the current interface view.
        """
        if self.label is None or self.probability is None:
            showinfo(title="PDF", message="Primero ejecute una predicción.")
            return

        capture = tkcap.CAP(self.root)
        image_name = f"Reporte{self.report_id}.jpg"
        pdf_name = f"Reporte{self.report_id}.pdf"

        capture.capture(image_name)

        with open(pdf_name, "wb") as pdf_file:
            pdf_file.write(img2pdf.convert(image_name))

        self.report_id += 1

        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def clear_interface(self) -> None:
        """
        Clear all interface fields and image previews.
        """
        answer = askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=WARNING,
        )

        if not answer:
            return

        self.entry_id.delete(0, "end")
        self.text_result.delete("1.0", END)
        self.text_probability.delete("1.0", END)
        self.image_box_original.delete("1.0", END)
        self.image_box_heatmap.delete("1.0", END)

        self.image_path = None
        self.label = None
        self.probability = None
        self.original_preview = None
        self.heatmap_preview = None

        self.button_predict["state"] = "disabled"

        showinfo(title="Borrar", message="Los datos se borraron con éxito.")


def main() -> int:
    """
    Run the graphical application.

    Returns:
        Exit status code.
    """
    App()
    return 0


if __name__ == "__main__":
    main()