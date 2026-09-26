
import os
import tempfile
import uuid
import zipfile
import io
import re
from fastapi import UploadFile


# ---------------- FILE TYPE ----------------
def get_file_type(filename: str):
    ext = os.path.splitext(filename)[-1].lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    if ext in image_exts:
        return 'image'
    elif ext == '.pdf':
        return 'pdf'
    else:
        return ext.lstrip('.')


# ---------------- LANGUAGE DETECTION ----------------
def detect_language(text: str) -> str:
    """
    Detect Hindi / Marathi / English using simple heuristics.
    """
    if not text:
        return "unknown"

    if re.search(r'[\u0900-\u097F]', text):
        if any(word in text for word in ["आहे", "म्हणजे", "तुम्ही", "काय"]):
            return "mr"
        return "hi"

    return "en"


# ---------------- FILENAME CLEANER ----------------
def safe_filename(filename: str) -> str:
    """
    Make filename safe while preserving multilingual characters.
    """
    filename = filename.strip()

    # Remove dangerous characters but keep Unicode
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    return filename


# ---------------- SAVE FILE TEMP ----------------
def save_upload_file_tmp(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename)[-1]

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path.write(upload_file.file.read())
    tmp_path.close()

    return tmp_path.name


# ---------------- MAIN PROCESSOR ----------------
def read_and_prepare_file(file: UploadFile, tmp_dir: str):
    """
    Accepts UploadFile (zip/pdf/image), extracts and saves files to disk.

    Returns:
    [
        {
            'filename': ...,
            'local_file_path': ...,
            'file_size': ...,
            'ext': ...,
            'file_uuid': ...,
            'file_bytes': ...,
            'detected_language': ...
        },
        ...
    ]
    """

    result = []
    filename = safe_filename(file.filename)
    ext = os.path.splitext(filename)[-1].lower()

    allowed_image_exts = [".jpg", ".jpeg", ".png"]
    allowed_pdf_exts = [".pdf"]
    allowed_zip_exts = [".zip"]

    os.makedirs(tmp_dir, exist_ok=True)

    # ---------------- ZIP FILE ----------------
    if ext in allowed_zip_exts:
        with zipfile.ZipFile(io.BytesIO(file.file.read())) as zip_ref:
            for member in zip_ref.namelist():

                member_name = safe_filename(os.path.basename(member))

                if not member_name.lower().endswith(tuple(allowed_pdf_exts + allowed_image_exts)):
                    continue

                file_uuid = str(uuid.uuid4())
                base_folder = os.path.join(tmp_dir, file_uuid)
                os.makedirs(base_folder, exist_ok=True)

                with zip_ref.open(member) as f:
                    file_bytes = f.read()

                local_file_path = os.path.abspath(os.path.join(base_folder, member_name))

                with open(local_file_path, "wb") as out_f:
                    out_f.write(file_bytes)

                file_size = os.path.getsize(local_file_path)
                member_ext = os.path.splitext(member_name)[-1].lower()

                # Try detecting language from filename (optional)
                detected_language = detect_language(member_name)

                result.append({
                    'filename': member_name,
                    'local_file_path': local_file_path,
                    'file_size': file_size,
                    'ext': member_ext,
                    'file_uuid': file_uuid,
                    'file_bytes': file_bytes,
                    'detected_language': detected_language
                })

    # ---------------- SINGLE FILE ----------------
    elif ext in allowed_pdf_exts + allowed_image_exts:
        file_uuid = str(uuid.uuid4())
        base_folder = os.path.join(tmp_dir, file_uuid)
        os.makedirs(base_folder, exist_ok=True)

        file_bytes = file.file.read()
        local_file_path = os.path.abspath(os.path.join(base_folder, filename))

        with open(local_file_path, "wb") as out_f:
            out_f.write(file_bytes)

        file_size = os.path.getsize(local_file_path)

        detected_language = detect_language(filename)

        result.append({
            'filename': filename,
            'local_file_path': local_file_path,
            'file_size': file_size,
            'ext': ext,
            'file_uuid': file_uuid,
            'file_bytes': file_bytes,
            'detected_language': detected_language
        })

    return result
