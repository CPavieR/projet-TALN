import math
import requests
import json
import re
import ast
import copy
import csv
from lib_helpers import HelperJDM
import requests_cache
import time

# Configuration API
link_api = "https://jdm-api.demo.lirmm.fr/schema"
api_get_node_by_name = "https://jdm-api.demo.lirmm.fr/v0/node_by_name/{node_name}"
get_relation_from = "https://jdm-api.demo.lirmm.fr/v0/relations/from/{node1_name}"
get_relation_between = "https://jdm-api.demo.lirmm.fr/v0/relations/from/{node1_name}/to/{node2_name}"
get_node_by_id = "https://jdm-api.demo.lirmm.fr/v0/node_by_id/{node_id}"
cache = {}

requests_cache.install_cache('jdm_cache', backend='sqlite', expire_after=None)
session = requests.Session()

def translate_relationNBtoNOM(relation):
    nom = "Unknown"
    try:
        nom = HelperJDM.nombre_a_nom[relation]
        return nom
    except Exception:
        return nom

def requestWrapper(url):
    global cache
    response = session.get(url)
    return response.text

def getNodeByName(node_name):
    jsonString = requestWrapper(
        api_get_node_by_name.format(node_name=node_name))
    return json.loads(jsonString)

def directRelation(node1, node2, wanted_relation):
    li_relation = requestWrapper(get_relation_between.format(
        node1_name=node1["name"], node2_name=node2["name"]))
    li_relation = json.loads(li_relation)
    li_relation["relations"] = [
        relation for relation in li_relation["relations"] if relation["type"] == wanted_relation]
    return li_relation

def get_refinements(term, relation_type='r_raff_sem'):
    url = f"https://jdm-api.demo.lirmm.fr/v0/refinements/{term}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status code {response.status_code}")
        return None

def getAllRelationsBetween(node1_name, node2_name):
    """Récupère toutes les relations entre deux nœuds"""
    try:
        url = get_relation_between.format(node1_name=node1_name, node2_name=node2_name)
        response = session.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("relations", [])
        else:
            print(f"Erreur API pour {node1_name} -> {node2_name}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Erreur lors de la récupération des relations: {e}")
        return []

def analyzeCorpus(tsv_file_path):
    """
    Analyse le corpus TSV et récupère les relations JDM pour chaque paire
    """
    results = []
    
    # Mapping des relations du corpus vers les IDs JDM
    relation_mapping = HelperJDM.nom_a_nombre
    
    # Lire le corpus TSV
    with open(tsv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        for i, row in enumerate(reader):
            if i % 10 == 0:
                print(f"Traitement ligne {i}...")
            
            relation_semantique = row['relation_semantique']
            forme_complete = row['forme_complete']
            gn1 = row['GN1']
            gn2 = row['GN2']
            
            # Récupérer toutes les relations entre GN1 et GN2
            all_relations = getAllRelationsBetween(gn1, gn2)
            
            # Filtrer et formater les relations trouvées
            found_relations = []
            expected_relation_found = False
            expected_relation_id = relation_mapping.get(relation_semantique)
            
            for rel in all_relations:
                rel_type = rel.get('type')
                rel_weight = rel.get('w', 0)
                rel_name = translate_relationNBtoNOM(rel_type)
                
                found_relations.append({
                    'id': rel_type,
                    'name': rel_name,
                    'weight': rel_weight
                })
                
                if rel_type == expected_relation_id:
                    expected_relation_found = True
            
            # Aussi vérifier les relations inverses (GN2 -> GN1)
            all_relations_inverse = getAllRelationsBetween(gn2, gn1)
            found_relations_inverse = []
            
            for rel in all_relations_inverse:
                rel_type = rel.get('type')
                rel_weight = rel.get('weight', 0)
                rel_name = translate_relationNBtoNOM(rel_type)
                
                found_relations_inverse.append({
                    'id': rel_type,
                    'name': rel_name,
                    'weight': rel_weight
                })
            
            result = {
                'forme_complete': forme_complete,
                'gn1': gn1,
                'gn2': gn2,
                'relation_attendue': relation_semantique,
                'relation_attendue_id': expected_relation_id,
                'relation_trouvee': expected_relation_found,
                'relations_jdm_directes': found_relations,
                'relations_jdm_inverses': found_relations_inverse,
                'nb_relations_directes': len(found_relations),
                'nb_relations_inverses': len(found_relations_inverse)
            }
            
            results.append(result)
            
            # Petit délai pour ne pas surcharger l'API
            time.sleep(0.1)
    
    return results

def generateReport(results):
    """
    Génère un rapport d'analyse du corpus
    """
    print("\n" + "="*80)
    print("RAPPORT D'ANALYSE DU CORPUS")
    print("="*80 + "\n")
    
    # Statistiques globales
    total = len(results)
    found_count = sum(1 for r in results if r['relation_trouvee'])
    not_found_count = total - found_count
    
    print(f"Total d'exemples analysés: {total}")
    print(f"Relations attendues trouvées dans JDM: {found_count} ({found_count/total*100:.1f}%)")
    print(f"Relations attendues non trouvées: {not_found_count} ({not_found_count/total*100:.1f}%)")
    
    # Statistiques par type de relation
    print("\n" + "-"*40)
    print("ANALYSE PAR TYPE DE RELATION")
    print("-"*40 + "\n")
    
    relation_stats = {}
    for result in results:
        rel_type = result['relation_attendue']
        if rel_type not in relation_stats:
            relation_stats[rel_type] = {'total': 0, 'found': 0, 'examples_not_found': []}
        
        relation_stats[rel_type]['total'] += 1
        if result['relation_trouvee']:
            relation_stats[rel_type]['found'] += 1
        else:
            relation_stats[rel_type]['examples_not_found'].append(result['forme_complete'])
    
    for rel_type, stats in sorted(relation_stats.items()):
        percentage = stats['found'] / stats['total'] * 100
        print(f"{rel_type}:")
        print(f"  Trouvées: {stats['found']}/{stats['total']} ({percentage:.1f}%)")
        if stats['examples_not_found'] and len(stats['examples_not_found']) <= 3:
            print(f"  Non trouvées: {', '.join(stats['examples_not_found'][:3])}")
    
    # Exemples de relations multiples
    print("\n" + "-"*40)
    print("EXEMPLES AVEC RELATIONS MULTIPLES")
    print("-"*40 + "\n")
    
    multi_relations = [r for r in results if r['nb_relations_directes'] > 1][:5]
    for result in multi_relations:
        print(f"\n'{result['forme_complete']}' ({result['gn1']} -> {result['gn2']}):")
        print(f"  Attendue: {result['relation_attendue']}")
        print(f"  Trouvées dans JDM:")
        for rel in result['relations_jdm_directes']:
            print(f"    - {rel['name']} (poids: {rel['weight']})")
    
    return relation_stats

def saveResults(results, output_file="corpus_analysis_results.json"):
    """
    Sauvegarde les résultats dans un fichier JSON
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats sauvegardés dans {output_file}")

# Fonction principale
def main():
    # Remplacez par le chemin de votre fichier TSV
    tsv_file = "corpus_genitif_tsv.tsv"
    
    print("Début de l'analyse du corpus...")
    print("Cela peut prendre plusieurs minutes selon la taille du corpus.\n")
    
    # Analyser le corpus
    results = analyzeCorpus(tsv_file)
    
    # Générer le rapport
    stats = generateReport(results)
    
    # Sauvegarder les résultats
    saveResults(results)
    
    # Optionnel: Sauvegarder aussi un CSV pour analyse dans Excel
    with open("corpus_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['forme_complete', 'gn1', 'gn2', 'relation_attendue', 
                     'relation_trouvee', 'nb_relations_directes', 'nb_relations_inverses']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    
    print("\nAnalyse terminée!")
    print("Fichiers générés:")
    print("  - corpus_analysis_results.json (résultats détaillés)")
    print("  - corpus_analysis.csv (résumé pour tableur)")

if __name__ == "__main__":
    main()