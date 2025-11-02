import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# ----------------------------------------------------
# ✅ Load Model and Tokenizer
# ----------------------------------------------------
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# ✅ Language code mapping (50 languages)
LANG_CODES = {
    "English": "en_XX", "Hindi": "hi_IN", "French": "fr_XX", "Spanish": "es_XX",
    "German": "de_DE", "Italian": "it_IT", "Russian": "ru_RU", "Chinese": "zh_CN",
    "Japanese": "ja_XX", "Korean": "ko_KR", "Arabic": "ar_AR", "Portuguese": "pt_XX",
    "Turkish": "tr_TR", "Dutch": "nl_XX", "Polish": "pl_PL", "Indonesian": "id_ID",
    "Vietnamese": "vi_VN", "Thai": "th_TH", "Romanian": "ro_RO", "Greek": "el_GR",
    "Czech": "cs_CZ", "Swedish": "sv_SE", "Finnish": "fi_FI", "Danish": "da_DK",
    "Bulgarian": "bg_BG", "Hungarian": "hu_HU", "Lithuanian": "lt_LT", "Latvian": "lv_LV",
    "Estonian": "et_EE", "Croatian": "hr_HR", "Malay": "ms_MY", "Hebrew": "he_IL",
    "Urdu": "ur_PK", "Tamil": "ta_IN", "Telugu": "te_IN", "Kannada": "kn_IN",
    "Gujarati": "gu_IN", "Bengali": "bn_IN", "Marathi": "mr_IN", "Punjabi": "pa_IN",
    "Sinhala": "si_LK", "Swahili": "sw_KE", "Ukrainian": "uk_UA", "Serbian": "sr_RS",
    "Norwegian": "no_XX", "Slovak": "sk_SK", "Icelandic": "is_IS", "Persian": "fa_IR",
    "Tagalog": "tl_XX", "Afrikaans": "af_ZA"
}

# ----------------------------------------------------
# ✅ Translation Function
# ----------------------------------------------------
def translate_text(text, src_lang, tgt_lang):
    if not text.strip():
        return "⚠️ Please enter some text to translate."
    tokenizer.src_lang = LANG_CODES[src_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANG_CODES[tgt_lang]],
        max_length=200
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ----------------------------------------------------
# ✅ Swap Function
# ----------------------------------------------------
def swap_languages(src, tgt):
    return tgt, src

# ----------------------------------------------------
# ✅ UI Design
# ----------------------------------------------------
with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="purple", secondary_hue="violet")) as demo:
    gr.Markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #b794f4; font-size: 3em; font-weight: bold; margin-bottom: 0.2em;'>
            🌐 TalkBuddy
        </h1>
        <p style='color: #d6bcfa; font-size: 1.2em; margin-top: 0;'>
            “Language is the bridge that connects hearts across cultures 🌍”
        </p>
        <hr style='border: 1px solid #805ad5; width: 60%; margin: 15px auto;'>
    </div>
    """)

    with gr.Row():
        text_input = gr.Textbox(
            label="Enter text",
            placeholder="Type something to translate...",
            lines=4
        )

    with gr.Row():
        src_lang = gr.Dropdown(
            choices=list(LANG_CODES.keys()),
            label="Source Language",
            value="English"
        )
        swap_btn = gr.Button("🔁 Swap", elem_id="swap_button")
        tgt_lang = gr.Dropdown(
            choices=list(LANG_CODES.keys()),
            label="Target Language",
            value="Hindi"
        )

    translate_btn = gr.Button("✨ Translate", variant="primary", elem_id="translate_button")
    output = gr.Textbox(label="Translation", lines=4)

    # ✅ Actions
    translate_btn.click(fn=translate_text, inputs=[text_input, src_lang, tgt_lang], outputs=output)
    swap_btn.click(fn=swap_languages, inputs=[src_lang, tgt_lang], outputs=[src_lang, tgt_lang])

    # ✅ Small CSS tweaks for beauty
    gr.HTML("""
    <style>
        #translate_button {
            background: linear-gradient(90deg, #805ad5, #b794f4);
            color: white;
            font-weight: bold;
            transition: 0.3s ease-in-out;
        }
        #translate_button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px #b794f4;
        }
        #swap_button {
            background: #6b46c1;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        #swap_button:hover {
            background: #9f7aea;
            transform: scale(1.1);
        }
        footer {visibility: hidden;}
    </style>
    """)

# ----------------------------------------------------
# ✅ Launch the app
# ----------------------------------------------------
demo.launch()
