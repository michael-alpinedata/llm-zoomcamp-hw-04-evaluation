import os
import glob
import nbformat
from pathlib import Path

def normalize_code(code_str):
    if not code_str: return ""
    return '\n'.join([line.strip() for line in code_str.split('\n') if line.strip()])

def parse_markdown_files(md_directory):
    md_files = sorted(glob.glob(os.path.join(md_directory, "*.md")))
    elements = []
    
    for filepath in md_files:
        in_code_block = False
        current_code = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if stripped == '```python':
                    in_code_block, current_code = True, []
                elif stripped == '```' and in_code_block:
                    in_code_block = False
                    elements.append(('code', '\n'.join(current_code)))
                elif in_code_block:
                    current_code.append(line.rstrip('\n'))
                elif line.startswith('#'):
                    elements.append(('header', stripped))
    
    tasks = []
    current_headers = []
    for i, (etype, text) in enumerate(elements):
        if etype == 'header':
            current_headers.append(text)
        elif etype == 'code' and current_headers:
            # On stocke code_1 et le bloc suivant (code_2) s'il existe
            code_1 = text
            code_2 = next((elements[j][1] for j in range(i + 1, len(elements)) if elements[j][0] == 'code'), None)
            tasks.append({'headers': current_headers, 'code_1': code_1, 'code_2': code_2})
            current_headers = []
    return tasks

def find_code_anchor(nb_cells, start_idx, code_to_find):
    if not code_to_find: return None
    target = normalize_code(code_to_find)
    code_cells = [(idx, cell) for idx, cell in enumerate(nb_cells) if cell.cell_type == 'code']
    for idx, cell in code_cells:
        if idx >= start_idx and normalize_code(cell.source) == target:
            return idx
    return None

def process_notebooks(tasks, nb_directory, output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    nb_files = sorted(glob.glob(os.path.join(nb_directory, "*.ipynb")))
    notebooks = [{'path': f, 'nb': nbformat.read(f, as_version=4), 'name': Path(f).name} for f in nb_files]
    
    if not notebooks:
        print("Aucun notebook trouvé.")
        return []

    active_nb_idx = 0
    start_cell_idx = 0
    deferred_tasks = []  # File d'attente pour les titres temporairement non placés
    failures = []

    for task in tasks:
        found = False
        
        # On tente de trouver une ancre avec code_1 puis code_2
        for anchor_code in [task['code_1'], task['code_2']]:
            if not anchor_code: 
                continue
            
            # 1. Recherche dans le notebook actuel
            match = find_code_anchor(notebooks[active_nb_idx]['nb'].cells, start_cell_idx, anchor_code)
            if match is not None:
                # ÉTAPE DE RÉCUPÉRATION : Si on avait des titres en attente, on les distribue proportionnellement
                if deferred_tasks:
                    k = len(deferred_tasks)
                    start = start_cell_idx
                    end = match
                    # Calcul des points d'insertion théoriques (répartition uniforme)
                    insertion_points = [start + round((end - start) * (i / (k + 1))) for i in range(1, k + 1)]
                    
                    # Insertion de droite à gauche pour éviter que le décalage des index ne fausse les positions
                    for task_to_ins, ins_idx in reversed(list(zip(deferred_tasks, insertion_points))):
                        md_source = '\n\n'.join(task_to_ins['headers'])
                        new_cell = nbformat.v4.new_markdown_cell(md_source)
                        notebooks[active_nb_idx]['nb'].cells.insert(ins_idx, new_cell)
                    
                    # Le match actuel a été poussé vers le bas de 'k' cellules
                    match += k
                    deferred_tasks = [] # Réinitialisation de la file d'attente

                # Insertion du titre actuel qui a réussi son ancrage
                md_source = '\n\n'.join(task['headers'])
                new_cell = nbformat.v4.new_markdown_cell(md_source)
                notebooks[active_nb_idx]['nb'].cells.insert(match, new_cell)
                
                start_cell_idx = match + 2
                found = True
                break
            
            # 2. Recherche dans le notebook suivant immédiat (Saut unique)
            if active_nb_idx + 1 < len(notebooks):
                match_next = find_code_anchor(notebooks[active_nb_idx + 1]['nb'].cells, 0, anchor_code)
                if match_next is not None:
                    # Avant de changer de notebook, on vide les titres en attente à la fin du notebook actuel
                    if deferred_tasks:
                        k = len(deferred_tasks)
                        start = start_cell_idx
                        end = len(notebooks[active_nb_idx]['nb'].cells)
                        insertion_points = [start + round((end - start) * (i / (k + 1))) for i in range(1, k + 1)]
                        
                        for task_to_ins, ins_idx in reversed(list(zip(deferred_tasks, insertion_points))):
                            md_source = '\n\n'.join(task_to_ins['headers'])
                            new_cell = nbformat.v4.new_markdown_cell(md_source)
                            notebooks[active_nb_idx]['nb'].cells.insert(ins_idx, new_cell)
                        deferred_tasks = []

                    # Changement officiel de contexte vers le notebook suivant
                    active_nb_idx += 1
                    
                    # Insertion du titre actuel dans le nouveau notebook
                    md_source = '\n\n'.join(task['headers'])
                    new_cell = nbformat.v4.new_markdown_cell(md_source)
                    notebooks[active_nb_idx]['nb'].cells.insert(match_next, new_cell)
                    
                    start_cell_idx = match_next + 2
                    found = True
                    break
        
        # Si aucune cellule de code (ni code_1 ni code_2) n'a été trouvée nulle part
        if not found:
            deferred_tasks.append(task)

    # GESTION DES RESCAPÉS FINAUX : Si des titres restent en attente à la toute fin du script
    # (aucun titre suivant n'a pu être validé pour servir de borne de fin)
    if deferred_tasks:
        for leftover_task in deferred_tasks:
            failures.append({'task': leftover_task, 'nb_name': notebooks[active_nb_idx]['name']})

    # Sauvegarde des fichiers
    for item in notebooks:
        out_path = os.path.join(output_directory, item['name'])
        nbformat.write(item['nb'], out_path)
        
    return failures

def generate_report(failures, output_directory):
    """Génère le rapport Markdown pour les insertions en échec."""
    if not failures:
        return
        
    report_path = os.path.join(output_directory, "rapport_insertions.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Rapport des insertions en échec\n\n")
        f.write("Les titres suivants n'ont pas pu être alignés automatiquement (ni par correspondance d'ancre, ni par calcul d'interpolation).\n\n")
        
        for fail in failures:
            f.write("---\n\n")
            f.write(f"### ❌ Titre(s) rejeté(s) :\n")
            for h in fail['task']['headers']:
                f.write(f"- `{h}`\n")
                
            f.write(f"\n**Recherché à partir du notebook :** `{fail['nb_name']}`\n\n")
            
            f.write("**Ancre principale recherchée (code_1) :**\n")
            f.write("```python\n")
            f.write(fail['task']['code_1'] + "\n")
            f.write("```\n\n")
            
            if fail['task']['code_2']:
                f.write("**Ancre secondaire recherchée (code_2) :**\n")
                f.write("```python\n")
                f.write(fail['task']['code_2'] + "\n")
                f.write("```\n\n")
            else:
                f.write("**Ancre secondaire (code_2) :** *Aucune (fin de document)*\n\n")

if __name__ == "__main__":
    # Configuration des dossiers (à adapter selon votre structure)
    MD_DIR = "../lessons_processed"
    NB_DIR = "./"
    OUT_DIR = "./notebooks_finaux"
    
    print("Analyse des fichiers Markdown...")
    tasks = parse_markdown_files(MD_DIR)
    
    print(f"{len(tasks)} groupes de titres identifiés. Début de l'injection (ancre unique)...")
    failures = process_notebooks(tasks, NB_DIR, OUT_DIR)
    
    if failures:
        generate_report(failures, OUT_DIR)
        print(f"Terminé avec {len(failures)} échec(s). Consultez '{OUT_DIR}/rapport_insertions.md'.")
    else:
        print("Terminé avec succès. Aucun échec d'insertion.")

