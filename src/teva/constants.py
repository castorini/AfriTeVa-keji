from typing import Final

PRETRAINING_LANGUAGES: Final = frozenset([ 
    "afr", "amh", "arz", "eng", 
    "fra", "hau", "ibo", "kin", 
    "mlg", "nya", "orm", "por",
    "sna", "som", "sot", "swa",
    "tir", "xho", "yor", "zul"
])

LAFAND_FR_PIVOT_LANGUAGES: Final = frozenset([])
LAFAND_EN_PIVOT_LANGUAGES: Final = frozenset([
    "hau", "pcm", "swa", "ibo", "yor", "zul", "tsn", "twi", # TODO: @theyorubayesian: Include xho, zul
])

MASAKHANEWS_LANGUAGES: Final = frozenset([
    "amh", "eng", "fra", "hau",
    "ibo", "lin", "lug", "orm", 
    "pcm", "run", "sna", "som", 
    "swa", "tir", "xho", "yor"
])

XLSUM_LANGUAGES: Final = frozenset([
    "amharic", "arabic", "english", "french", "hausa",
    "igbo", "kirundi", "oromo", "pidgin", "portuguese",
    "somali", "swahili", "tigrinya", "yoruba",
])

XP3X_LANGUAGE_CODES: Final = frozenset([
    "afr_Latn",
    "amh_Ethi", "aka_Latn", "bam_Latn", "bem_Latn", "ewe_Latn",
    "fon_Latn", "gaz_Latn", "hau_Latn", "ibo_Latn", "kam_Latn", "kik_Latn",
    "kin_Latn", "kmb_Latn", "knc_Arab", "knc_Latn", "kon_Latn", "lin_Latn",
    "lug_Latn", "luo_Latn", "nso_Latn", "nya_Latn", "pcm_Latn", "plt_Latn",
    "run_Latn", "sna_Latn", "som_Latn", "sot_Latn", "ssw_Latn", "swh_Latn",
    "tir_Ethi", "tsn_Latn", "tum_Latn", "twi_Latn", "umb_Latn", "wol_Latn",
    "xho_Latn", "yor_Latn", "zul_Latn",
    "aeb_Arab", "arb_Arab", "arb_Latn", "arq_Arab", "ary_Arab", "arz_Arab",
    "eng_Latn", "fra_Latn", "por_Latn"
])

AYA_HUMAN_LANGUAGES: Final = frozenset([
        "amharic", "egyptian_arabic", "english",
        "french", "hausa", "igbo", "nyanja",
        "plateau_malagasy", "portuguese", "shona",
        "somali", "swahili", "xhosa", "yoruba", "zulu",
        # African languages not in wura
        "moroccan_arabic", "wolof"
])

AYA_LANGUAGES: Final = frozenset([
    "afrikaans", "amharic", "egyptian_arabic", "english",
    "french", "hausa", "igbo", "kinyarwanda", "nyanja",
    "plateau_malagasy", "portuguese", "shona", "somali",
    "swahili", "southern_sotho", "xhosa", "yoruba", "zulu"
    "algerian_arabic", "moroccan_arabic", "tunisian_arabic",  # These are alt arabic forms we could support
    "mozambican_portuguese",                                  # Alt portuguese forms
    "bemba", "central_kanuri", "fon", "twi", "wolof"          # These are African languages not in WURA
])

AYA_TRANSLATED_DATASETS: Final = frozenset([
    "adversarial_qa_(t)","cnn-daily-mail_(t)", "dolly-v2_(t)",
    "flan-coqa_(t)", "flan-cot-submix_(t)", "flan-gem-wiki-lingua_(t)",
    "flan-lambada_(t)", "flan-unified-qa_(t)", "hotpotqa_(t)",
    "joke-explaination-inst_(t)", "mintaka-inst_(t)", "mlqa-en_(t)",
    "nq-open_(t)", "paws-wiki_(t)", "piqa_(t)", "soda-inst_(t)",
    "wiki_qa_(t)", "wiki-split-inst_(t)", "xlel_wd-inst_(t)"
])

AYA_TEMPLATED_DATASETS: Final = frozenset([
    "afriqa-inst", "afrisenti-inst", "amharic_qa",
    "joke-explaination-inst", "masakhanews-inst",
    "mintaka-inst", "ntx-llm-inst", "nusax-senti-inst",
    "scirepeval-biomimicry-inst", "soda-inst", "uner-llm-inst",
    "wiki-split-inst", "xlel_wd-inst", "xwikis-inst"
])
