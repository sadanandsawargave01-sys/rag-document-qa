
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import os
import shutil
from pathlib import Path
from pdf2image import convert_from_path

from google.oauth2 import service_account
from google.cloud import vision
from google.cloud.vision_v1 import types
from google.protobuf.json_format import MessageToDict

from config import GOOGLE_CREDS

class PDFProcessor:
    def __init__(self, file_location, file_uuid):
        self.file_location = file_location
        self.file_uuid = file_uuid

        self.BASE_FOLDER = f"tmp/{file_uuid}"
        self.SPLITED_FOLDER_PDF = f"{self.BASE_FOLDER}/pdf"
        self.SPLITED_FOLDER_IMG = f"{self.BASE_FOLDER}/img"

        os.makedirs(self.SPLITED_FOLDER_PDF, exist_ok=True)
        os.makedirs(self.SPLITED_FOLDER_IMG, exist_ok=True)

        # Google Vision Client
        self.google_credentials = service_account.Credentials.from_service_account_info(GOOGLE_CREDS)
        self.client = vision.ImageAnnotatorClient(credentials=self.google_credentials)

    # ---------------- CLEANUP ----------------
    def cleanup(self):
        try:
            if os.path.exists(self.BASE_FOLDER):
                shutil.rmtree(self.BASE_FOLDER)
                print(f"✅ Cleaned up: {self.BASE_FOLDER}")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")

    # ---------------- TEXT CLEANING ----------------
    def clean_text(self, text: str) -> str:
        """
        Basic OCR cleanup for multilingual text.
        """
        if not text:
            return ""

        # Normalize spaces & newlines
        text = text.replace("\n\n", "\n")
        text = " ".join(text.split())

        return text.strip()

    # ---------------- OCR ----------------
    def google_based_image_text_extraction(self, image_path: str) -> str:
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()

            content_image = types.Image(content=image_data)

            # 🔥 Multilingual hints (important!)
            image_context = vision.ImageContext(
                language_hints=["en", "hi", "mr"]  # English, Hindi, Marathi
            )

            response = self.client.document_text_detection(
                image=content_image,
                image_context=image_context
            )

            if response.error.message:
                raise Exception(response.error.message)

            document = response.full_text_annotation

            if document and document.text:
                return self.clean_text(document.text)

            return ""

        except Exception as e:
            raise Exception(f"Error processing image OCR: {e}")

    # ---------------- PDF PROCESS ----------------
    def process_pdf(self) -> list[str]:
        extracted_texts = []

        try:
            image_paths = self.convert_pdf_to_images()

            for image_path in image_paths:
                text = self.google_based_image_text_extraction(image_path)
                extracted_texts.append(text)

            return extracted_texts

        finally:
            self.cleanup()

    # ---------------- IMAGE PROCESS ----------------
    def process_image(self) -> list[str]:
        try:
            text = self.google_based_image_text_extraction(self.file_location)
            return [text]  # ✅ consistent return type (list)

        finally:
            self.cleanup()

    # ---------------- PDF → IMAGE ----------------
    def convert_pdf_to_images(self) -> list[str]:
        image_paths = []

        images = convert_from_path(
            self.file_location,
            dpi=300,              # 🔥 better OCR accuracy
            fmt='jpeg'
        )

        for i, image in enumerate(images):
            image_path = os.path.join(self.SPLITED_FOLDER_IMG, f"page-{i}.jpg")

            # Optional compression for speed
            image.save(image_path, "JPEG", quality=85)

            image_paths.append(image_path)

        return image_paths
