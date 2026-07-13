import os
import re
import argparse

def process_markdown_files(source_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    title_pattern = re.compile(r'^(#{1,2})\s+(.*)')
    
    for filename in os.listdir(source_dir):
        if filename.endswith(".md"):
            match_file = re.match(r'^(\d+)-', filename)
            if not match_file:
                continue
            
            file_num = int(match_file.group(1))
            
            source_path = os.path.join(source_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            with open(source_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            sub_counter = 0
            in_code_block = False
            
            for line in lines:
                # Gestion des blocs de code ```
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    new_lines.append(line)
                    continue
                
                # Si on est dans un bloc de code, on garde la ligne
                if in_code_block:
                    new_lines.append(line)
                    continue
                
                # Si on est hors code, on vérifie si c'est un titre
                match_title = title_pattern.match(line)
                if match_title:
                    level = match_title.group(1)
                    content = match_title.group(2)
                    
                    if level == "#":
                        new_lines.append(f"# {file_num} {content}\n")
                    elif level == "##":
                        sub_counter += 1
                        new_lines.append(f"## {file_num}.{sub_counter} {content}\n")
                
                # Note : Tout le texte brut hors code et hors titre est ignoré
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"Fichier filtré : {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrait titres et blocs de code uniquement.")
    parser.add_argument("source", help="Dossier source")
    parser.add_argument("output", help="Dossier destination")
    
    args = parser.parse_args()
    process_markdown_files(args.source, args.output)