
from config import GOOGLE_API_KEY
from prompts import PROMPT_QP_FORMAT_EXTRACT, PROMPT_QP_GEN
import google.generativeai as genai
import os
import datetime
from fpdf import FPDF
import traceback


output_dir = "qgen_output_files"
os.makedirs(output_dir, exist_ok=True)

genai.configure(api_key=GOOGLE_API_KEY)


# ---------------- PROMPT BUILDER ----------------
def build_format_prompt(target_language=None):
    base = f"""
{PROMPT_QP_FORMAT_EXTRACT}

IMPORTANT:
- The input may contain Hindi, Marathi, or English.
"""

    if target_language:
        base += f"- Extract and convert the format into {target_language}.\n"
    else:
        base += "- Preserve the original language.\n"

    return base


def build_generation_prompt(format_text, target_language=None):
    base = f"""
{PROMPT_QP_GEN.format(format_text)}

IMPORTANT:
- Generate a complete question paper.
"""

    if target_language:
        base += f"- Generate the question paper in {target_language}.\n"
    else:
        base += "- Preserve the language of the source content.\n"

    return base


# ---------------- FORMAT EXTRACTION ----------------
def llm_question_paper_format_extract(
    image_path: str = '',
    pdf_path: str = '',
    llm_model: str = "models/gemini-2.5-flash",
    temperature: float = 0.3,
    top_k: int = 50,
    top_p: float = 0.3,
    target_language: str | None = None
):
    try:
        generation_config = genai.GenerationConfig(
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        model = genai.GenerativeModel(
            model_name=llm_model,
            generation_config=generation_config
        )

        prompt_parts = [build_format_prompt(target_language)]

        if image_path:
            prompt_parts.append(genai.upload_file(path=image_path))

        if pdf_path:
            prompt_parts.append(genai.upload_file(path=pdf_path))

        if len(prompt_parts) > 1:
            response = model.generate_content(prompt_parts)
            extracted_text = response.text.replace('```', '').replace('json\n', '').strip()
            return 'done', extracted_text

        return 'failed', 'No input source provided.'

    except Exception as error:
        return 'failed', str(error)


# ---------------- PDF GENERATOR (Unicode Safe) ----------------
class UnicodePDF(FPDF):
    def header(self):
        pass


def create_pdf(content: str, output_path: str):
    pdf = UnicodePDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ⚠️ IMPORTANT: Use Unicode font (required for Hindi/Marathi)
    font_path = os.path.join(os.path.dirname(__file__), "NotoSans-Regular.ttf")

    if not os.path.exists(font_path):
        raise RuntimeError("Unicode font file 'NotoSans-Regular.ttf' not found.")

    pdf.add_font("Noto", "", font_path, uni=True)
    pdf.set_font("Noto", size=12)

    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf.output(output_path)


# ---------------- QUESTION PAPER GENERATOR ----------------
def llm_question_paper_generator(
    knowledge_ip_image_path_list: list = [],
    knowledge_ip_pdf_path: str = '',
    qp_format_pdf: str = '',
    qp_format_image: str = '',
    llm_model: str = "models/gemini-2.5-flash",
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.5,
    target_language: str | None = None
):
    try:
        filename = f"generated_{datetime.datetime.now().strftime('%d-%m-%Y__%H-%M-%S')}.pdf"
        output_file_path = os.path.join(output_dir, filename)

        # Step 1: Extract format
        qp_generate_format = llm_question_paper_format_extract(
            qp_format_image,
            qp_format_pdf,
            llm_model,
            temperature,
            top_k,
            top_p,
            target_language
        )

        if qp_generate_format[0] != 'done':
            return 'failed', 'Failed to extract format.'

        # Step 2: Generate paper
        generation_config = genai.GenerationConfig(
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        model = genai.GenerativeModel(
            model_name=llm_model,
            generation_config=generation_config
        )

        prompt_parts = [build_generation_prompt(qp_generate_format[1], target_language)]

        for path in knowledge_ip_image_path_list:
            prompt_parts.append(genai.upload_file(path=path))

        if knowledge_ip_pdf_path:
            prompt_parts.append(genai.upload_file(path=knowledge_ip_pdf_path))

        if len(prompt_parts) > 1:
            response = model.generate_content(prompt_parts)
            extracted_text = response.text.replace('```', '').replace('json\n', '').strip()

            # Step 3: Create multilingual-safe PDF
            create_pdf(extracted_text, output_file_path)

            return 'done', output_file_path

        return 'failed', 'No knowledge source provided.'

    except Exception as error:
        print(traceback.format_exc())
        return 'failed', str(error)
