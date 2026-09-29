from prompts import RELATION_SYSTEM
from llm_client import LLMClient
def main():
    system_prompt = RELATION_SYSTEM
    user_prompt = """Convert the medical text below into triples that are useful for answering medical questions. avoid generating triples that are not in the text. Output ONLY the JSON array.
    Format: [["subject", "predicate", "object"], ...]
    Predicates allowed: causes, treated_with, risk_factor, symptoms, diagnosed_by, complication, medication, prevents, contraindicated_with, associated_with, indicates, defined_as
    Subjects/objects: specific named entities only (e.g. "Ampicillin", "Pneumonia", "headache") — never generic words like "treatment", "medications", "patients".
    Each triple must have subject != object. For treated_with: (drug, treated_with, disease) not the reverse.

    Medical text:
    A shiny gray element with atomic symbol As, atomic number 33, and atomic weight 75. It occurs throughout the universe, mostly in the form of metallic arsenides. Most forms are toxic. According to the Fourth Annual Report on Carcinogens (NTP 85-002, 1985), arsenic and certain arsenic compounds have been listed as known carcinogens. (From Merck Index, 11th ed)
    Arsenic is a chemical element; it has the symbol As and atomic number 33. It is a metalloid and one of the pnictogens, and therefore shares many properties with its group 15 neighbors phosphorus and antimony. Arsenic is notoriously toxic. It occurs naturally in many minerals, usually in combination with sulfur and metals, but also as a pure elemental crystal. It has various allotropes, but only the grey form, which has a metallic appearance, is important to industry.
    The primary use of arsenic is in alloys of lead (for example, in car batteries and ammunition). Arsenic is also a common n-type dopant in semiconductor electronic devices, and a component of the III–V compound semiconductor gallium arsenide. Arsenic and its compounds, especially the trioxide, are used in the production of pesticides, treated wood products, herbicides, and insecticides. These applications are declining with the increasing recognition of the persistent toxicity of arsenic and its compounds.
    Arsenic containing compounds have been known since ancient times to be poisonous to humans. However, a few species of bacteria are able to use arsenic compounds as respiratory metabolites. Trace quantities of arsenic have been proposed to be an essential dietary element in rats, hamsters, goats, and chickens. Research has not been conducted to determine whether small amounts of arsenic may play a role in human metabolism. However, arsenic poisoning occurs in multicellular life if quantities are larger than needed. Arsenic contamination of groundwater is a problem that affects millions of people across the world.
    The United States Environmental Protection Agency states that all forms of arsenic are a serious risk to human health. The United States Agency for Toxic Substances and Disease Registry ranked arsenic number 1 in its 2001 prioritized list of hazardous substances at Superfund sites. Arsenic is classified as a group 1 carcinogen."""

    client = LLMClient("inclusionai/ling-3.0-flash")
    response = client._call_once(system=system_prompt,user=user_prompt,temperature=0.5)
    print(response)
if __name__ == "__main__":
    main()
