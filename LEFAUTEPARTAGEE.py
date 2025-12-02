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
import click


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

def getNodeById(node_id):
    jsonString = requestWrapper(
        get_node_by_id.format(node_id=node_id))
    return json.loads(jsonString)

def directRelation(node1, node2, wanted_relation):
    li_relation = requestWrapper(get_relation_between.format(
        node1_name=node1["name"], node2_name=node2["name"]))
    li_relation = json.loads(li_relation)
    li_relation["relations"] = [
        relation for relation in li_relation["relations"] if relation["type"] == wanted_relation]
    return li_relation

def infoRelationFiltered(node, wanted_relation):

    li_relation = requestWrapper(get_relation_from.format(
        node1_name=node["name"]))
    li_relation = json.loads(li_relation)
    #normalize  weights
    weigth_max = max([relation['w'] for relation in li_relation["relations"] if 'w' in relation], default=1)
    for relation in li_relation["relations"]:
        if 'w' in relation:
            relation['w'] = relation['w'] / weigth_max
    #print(li_relation["relations"])
    li_relation["relations"] = [
        relation for relation in li_relation["relations"] if relation["type"] in wanted_relation]
    #print(li_relation["relations"])
    return li_relation

def infoRelationTop10(node):

    li_relation = requestWrapper(get_relation_from.format(
        node1_name=node["name"]))
    li_relation = json.loads(li_relation)
    weigths = [
        relation['w'] for relation in li_relation["relations"] if 'w' in relation]
    weigths = sorted(weigths, reverse=True)[:10]
    li_relation["relations"] = [
        relation for relation in li_relation["relations"] if relation.get('w', 0) in weigths]
    #print(li_relation["relations"])
    weigth_max = max(weigths, default=1)
    for relation in li_relation["relations"]:
        if 'w' in relation:
            relation['w'] = relation['w'] / weigth_max
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

def analyzeCorpus(tsv_file_path, delay=0.1):
    """
    Analyse le corpus TSV et récupère les relations JDM pour chaque paire
    """
    results = []
    
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
            
            result = process_genetive(gn1, gn2, relation_semantique, forme_complete)
            
            results.append(result)
            
            # Petit délai pour ne pas surcharger l'API
            time.sleep(delay)
    
    return results

def process_genetive(gn1, gn2, relation_semantique, forme_complete):
                # Récupérer toutes les relations entre GN1 et GN2
            all_relations = getAllRelationsBetween(gn1, gn2)
            node1 = getNodeByName(gn1)
            node2 = getNodeByName(gn2)
            if not "error" in node1 and not "error" in node2:

                # Récupérer les informations sémantiques des nœuds
                info_sem__relation_1 = infoRelationFiltered(getNodeByName(gn1),  [HelperJDM.nom_a_nombre["r_isa"],HelperJDM.nom_a_nombre["r_infopot"]])
                info_sem__relation_2 = infoRelationFiltered(getNodeByName(gn2),  [HelperJDM.nom_a_nombre["r_isa"],HelperJDM.nom_a_nombre["r_infopot"]])
                
                
                info_sem__relation_1["relations"] = info_sem__relation_1["relations"] + infoRelationTop10(getNodeByName(gn1))["relations"]
                info_sem__relation_2["relations"] = info_sem__relation_2["relations"] + infoRelationTop10(getNodeByName(gn2))["relations"]
                
            else:
                info_sem__relation_1 = {'relations': []}
                info_sem__relation_2 = {'relations': []}
            # Filtrer et formater les relations trouvées
            found_relations = []
            info_sem_1 = []
            info_sem_2 = []
            expected_relation_found = False
            if relation_semantique != None:
                expected_relation_id = HelperJDM.nom_a_nombre.get(relation_semantique)
            else:
                expected_relation_id = None
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
            
            for info in info_sem__relation_1['relations']:
                rel_name = translate_relationNBtoNOM(info.get('type'))
                if (rel_name != "r_infopot" or getNodeById(info.get('node2', None)).get('name', '').startswith("_INFO-SEM")):
                    node2_name = getNodeById(info.get('node2', None)).get('name', None)
                    if not node2_name.startswith(":r") :
                        info_sem_1.append({
                            'id': info.get('type'),
                            'name': translate_relationNBtoNOM(info.get('type')),
                            'weight': info.get('w', 0),
                            'node2': info.get('node2', None),
                            'node2_name': node2_name
                        })
            for info in info_sem__relation_2['relations']:
                rel_name = translate_relationNBtoNOM(info.get('type'))
                if (rel_name != "r_infopot" or getNodeById(info.get('node2', None)).get('name', '').startswith("_INFO-SEM")):
                    node2_name = getNodeById(info.get('node2', None)).get('name', None)
                    if not node2_name.startswith(":r") :
                        info_sem_2.append({
                            'id': info.get('type'),
                            'name': translate_relationNBtoNOM(info.get('type')),
                            'weight': info.get('w', 0),
                            'node2': info.get('node2', None),
                            'node2_name': node2_name
                        })

            # Aussi vérifier les relations inverses (GN2 -> GN1)
            all_relations_inverse = getAllRelationsBetween(gn2, gn1)
            found_relations_inverse = []
            
            for rel in all_relations_inverse:
                rel_type = rel.get('type')
                rel_weight = rel.get('w', 0)
                rel_name = translate_relationNBtoNOM(rel_type)
                
                found_relations_inverse.append({
                    'id': rel_type,
                    'name': rel_name,
                    'weight': rel_weight
                })
            
            return {
                'forme_complete': forme_complete,
                'gn1': gn1,
                'gn2': gn2,
                'relation_attendue': relation_semantique,
                'relation_attendue_id': expected_relation_id,
                'relation_trouvee': expected_relation_found,
                'relations_jdm_directes': found_relations,
                'relations_jdm_inverses': found_relations_inverse,
                'nb_relations_directes': len(found_relations),
                'nb_relations_inverses': len(found_relations_inverse),
                'info_sem_gn1': info_sem_1,
                'info_sem_gn2': info_sem_2
            }

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

@click.command()
@click.option('--tsv-file', default='corpus_genitif_tsv.tsv', help='Path to TSV file.')
@click.option('--output-json', default='corpus_analysis_results.json', help='JSON output file.')
@click.option('--output-csv', default='corpus_analysis_results.csv', help='CSV output file.')
@click.option('--delay', default=0.1, type=float, help='Delay between API requests in seconds.')
def main(tsv_file, output_json, output_csv, delay):
    print("Début de l'analyse du corpus...")
    print("Cela peut prendre plusieurs minutes selon la taille du corpus.\n")

    results = analyzeCorpus(tsv_file, delay=delay)

    stats = generateReport(results)
    
    saveResults(results, output_json)
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['forme_complete', 'gn1', 'gn2', 'relation_attendue', 
                    'relation_trouvee', 'nb_relations_directes', 'nb_relations_inverses']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    
    print("\nAnalyse terminée!")
    print("Fichiers générés:")

if __name__ == "__main__":
    main()