import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import click

@dataclass
class Rule:
    """A rule is a pair of signatures with a relation type and per-symbol weights"""
    signature_left: Set[str]  # Signature for term A (gn1)
    signature_right: Set[str]  # Signature for term B (gn2)
    signature_left_weights: Dict[str, float] = field(default_factory=dict)
    signature_right_weights: Dict[str, float] = field(default_factory=dict)
    relation_type: str = "" # pour quelle rel est la règle
    weight: int = 1  # Number of fusion total
    
class vectorCreator:
    
    def __init__(self, 
                 use_hypernyms: bool = True,
                 use_trt: bool = True,
                 use_sst: bool = True,
                 use_definiteness: bool = False,
                 fusion_threshold: float = 0.5,
                 trim_rules: bool = False):
        """
        Args:
            use_hypernyms: Include hypernyms (r_isa) in signatures
            use_trt: Include Target Relation Types in signatures
            use_sst: Include Standard Semantic Types in signatures
            use_definiteness: Include definiteness markers
            fusion_threshold: Similarity threshold for merging rules (0.5 in paper)
            trim_rules: Only keep non-merged rules for faster classification
        """
        self.use_hypernyms = use_hypernyms
        self.use_trt = use_trt
        self.use_sst = use_sst
        self.use_definiteness = use_definiteness
        self.fusion_threshold = fusion_threshold
        self.trim_rules = trim_rules
        
        self.rules: List[Rule] = []
        self.rules_by_type: Dict[str, List[Rule]] = defaultdict(list)
        
    def extract_signature(self, term_data: Dict, term_name: str, is_gn2: bool = False) -> Set[str]:
        """
        Extract signature for a term based on JDM data
        
        Args:
            term_data: The preprocessed JSON data for an example
            term: The actual term string (gn1 or gn2)
            is_gn2: Whether this is the second term (for definiteness)
            
        Returns:
            Set of symbols representing the term's semantic signature
        """
        signature = set()
        term_position = 'gn2' if is_gn2 else 'gn1'
        # Ajouter terme luimême
        signature.add(term_name)
        with open("info_sem_debug.json", "w", encoding="utf-8") as debug_file:
            json.dump(term_data, debug_file, ensure_ascii=False, indent=2)
        # H - r_isa relation
        if self.use_hypernyms and f'info_sem_{term_name}' in term_data:
            info_sem = term_data[f'info_sem_{term_name}']
            if isinstance(info_sem, list):
                for rel in info_sem:
                    if rel.get('name') == 'r_isa' and 'node2' in rel:
                        # node2 contains the hypernym ID, we'd need to map to string
                        # For now, use a symbolic representation
                        signature.add(f"{rel['node2_name']}")
        
        # TRT
        if self.use_trt:
            # on veut les relation menant au terme
            info_sem = term_data[f'info_sem_{term_position}']
            if isinstance(info_sem, list):
                weight_list = [rel["weight"] for rel in info_sem if "weight" in rel]
                weight_list.sort(reverse=True)
                if len(weight_list) > 10:
                    weight_list = weight_list[:10]
                seuil = weight_list[-1] if weight_list else 0
                for rel in info_sem:
                    if rel.get('weight') >= seuil:
                        signature.add(f"{rel.get('name')}")
                        

        #TODO : not in preprocess data yet
        # Extract Standard Semantic Types (SST)
        if self.use_sst and f'info_sem_{term_position}' in term_data:
            info_sem = term_data[f'info_sem_{term_position}']
            if isinstance(info_sem, list):
                for rel in info_sem:
                    if 'name' in rel and rel['name'].startswith('r_infopot'):
                        signature.add(rel['name'])
        
        # Definiteness (for gn2 only)
        if self.use_definiteness and is_gn2:
            #check if "de" is followed by article
            forme = term_data.get('forme_complete', '')
            if ' du ' in forme or ' de la ' in forme or ' des ' in forme or " de l'" in forme or " de " in forme:
                signature.add('DEF')
            if ' de ' in forme or " d'" in forme or " des ":
                signature.add('NODEF')
        
        return signature
    
    def cosine_similarity(self, sig1: Set[str] or Dict[str, float], sig2: Set[str] or Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two signatures (supports sets or dicts of weights)
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Cosine similarity score [0, 1]
        """
        #normalize inputs to dicts of weights
        def to_weight_dict(s):
            if isinstance(s, dict):
                return s
            if isinstance(s, set):
                return {k: 1.0 for k in s}
            return {}
        
        d1 = to_weight_dict(sig1)
        d2 = to_weight_dict(sig2)
        
        if not d1 or not d2:
            return 0.0
        
        # dot product
        dot = sum(d1.get(k, 0.0) * d2.get(k, 0.0) for k in d1.keys())
        magnitude1 = np.sqrt(sum(v * v for v in d1.values()))
        magnitude2 = np.sqrt(sum(v * v for v in d2.values()))
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot / (magnitude1 * magnitude2)
    
    def can_merge(self, rule1: Rule, rule2: Rule) -> bool:
        """
        Check if two rules can be merged based on similarity threshold
        
        Args:
            rule1: First rule
            rule2: Second rule
            
        Returns:
            True if rules should be merged
        """
        if rule1.relation_type != rule2.relation_type:
            return False
        
        # use per-symbol weight dicts for similarity
        sim_left = self.cosine_similarity(rule1.signature_left_weights, rule2.signature_left_weights)
        sim_right = self.cosine_similarity(rule1.signature_right_weights, rule2.signature_right_weights)
        
        avg_similarity = (sim_left + sim_right) / 2
        
        return avg_similarity >= self.fusion_threshold
    
    def merge_rules(self, rule1: Rule, rule2: Rule) -> Rule:
        """
        Merge two rules by taking the union of their signatures and summing per-symbol weights
        
        Args:
            rule1: First rule
            rule2: Second rule
            
        Returns:
            Merged rule
        """
        new_left = rule1.signature_left | rule2.signature_left
        new_right = rule1.signature_right | rule2.signature_right
        
        # merge weight dicts (sum weights for same keys)
        left_weights = rule1.signature_left_weights.copy()
        for k, v in rule2.signature_left_weights.items():
            left_weights[k] = left_weights.get(k, 0.0) + v
        
        right_weights = rule1.signature_right_weights.copy()
        for k, v in rule2.signature_right_weights.items():
            right_weights[k] = right_weights.get(k, 0.0) + v
        
        return Rule(
            signature_left=new_left,
            signature_right=new_right,
            signature_left_weights=left_weights,
            signature_right_weights=right_weights,
            relation_type=rule1.relation_type,
            weight=rule1.weight + rule2.weight
        )
    
    def aggregate_rules(self, rules: List[Rule]) -> List[Rule]:
        """
        fusion operations
        
        Args:
            rules: List of rules to aggregate
            
        Returns:
            Aggregated list of rules
        """
        if not rules:
            return []
        
        changed = True
        current_rules = rules.copy()
        
        while changed:
            changed = False
            new_rules = []
            merged_indices = set()
            
            for i, rule1 in enumerate(current_rules):
                if i in merged_indices:
                    continue
                
                merged = False
                for j, rule2 in enumerate(current_rules[i+1:], start=i+1):
                    if j in merged_indices:
                        continue
                    
                    if self.can_merge(rule1, rule2):
                        merged_rule = self.merge_rules(rule1, rule2)
                        new_rules.append(merged_rule)
                        merged_indices.add(i)
                        merged_indices.add(j)
                        merged = True
                        changed = True
                        break
                
                if not merged:
                    new_rules.append(rule1)
            
            current_rules = new_rules
        
        return current_rules
    
    def train(self, training_data: List[Dict]):
        """
        
        Args:
            training_data: List of preprocessed JSON examples
        """
        
        # Step 1: Create initial rules (one per example)
        initial_rules = []
        
        for example in training_data:
            gn1 = example.get('gn1', '')
            gn2 = example.get('gn2', '')
            relation = example.get('relation_attendue', '')
            
            # Extract signatures for both terms
            sig_left = self.extract_signature(example, gn1, is_gn2=False)
            sig_right = self.extract_signature(example, gn2, is_gn2=True)
            
            # create per-symbol weight dicts (default 1.0)
            left_weights = {s: 1.0 for s in sig_left}
            right_weights = {s: 1.0 for s in sig_right}
            
            rule = Rule(
                signature_left=sig_left,
                signature_right=sig_right,
                signature_left_weights=left_weights,
                signature_right_weights=right_weights,
                relation_type=relation,
                weight=1
            )
            
            initial_rules.append(rule)
        

        rules_by_type = defaultdict(list)
        for rule in initial_rules:
            rules_by_type[rule.relation_type].append(rule)
        

        all_aggregated_rules = []
        
        for relation_type, rules in rules_by_type.items():
            aggregated = self.aggregate_rules(rules)
            print(f"  {relation_type}: {len(rules)} -> {len(aggregated)} rules")
            all_aggregated_rules.extend(aggregated)
            self.rules_by_type[relation_type] = aggregated
            self.rules = self.rules+aggregated
        
        """
        if self.trim_rules:
            # Only keep rules that weren't merged (weight == 1) or are the result of merging
            trimmed = [r for r in all_aggregated_rules if r.weight > 1]
            print(f"Trimmed rules: {len(all_aggregated_rules)} -> {len(trimmed)}")
            self.rules = trimmed
        else:
            self.rules = all_aggregated_rules"""
        
        print(f"Training complete. Total rules: {len(self.rules)}")
    
    def classify(self, example: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        
        Args:
            example: Preprocessed JSON example
            
        Returns:
            Tuple of (predicted_relation, confidence, all_scores)
        """
        gn1 = example.get('gn1', '')
        gn2 = example.get('gn2', '')
        
        # extract signatures 
        sig_left = self.extract_signature(example, gn1, is_gn2=False)
        sig_right = self.extract_signature(example, gn2, is_gn2=True)
        
        scores = defaultdict(list)
        
        for type,rules in self.rules_by_type.items():
            for rule in rules:
                # compare  signatures
                sim_left = self.cosine_similarity(sig_left, rule.signature_left_weights)
                sim_right = self.cosine_similarity(sig_right, rule.signature_right_weights)
                
                # average similarity
                avg_sim = (sim_left + sim_right) / 2
                
                #weight by rule fusion weight
                weighted_score = avg_sim * rule.weight
                
                scores[rule.relation_type].append(weighted_score)
        
        #aggregate scores per relation type (max)
        final_scores = {rel: max(score_list) for rel, score_list in scores.items()}
        
        if not final_scores:
            return "UNKNOWN", 0.0, {}
        
        #get best prediction
        best_relation = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_relation]
        
        return best_relation, best_score, final_scores
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate on test dataset
        
        Args:
            test_data: List of test examples
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        total = len(test_data)
        
        #per-type metrics
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)
        
        predictions = []
        
        for example in test_data:
            expected = example.get('relation_attendue', '')
            predicted, confidence, all_scores = self.classify(example)
            
            predictions.append({
                'forme_complete': example.get('forme_complete', ''),
                'gn1': example.get('gn1', ''),
                'gn2': example.get('gn2', ''),
                'expected': expected,
                'predicted': predicted,
                'confidence': confidence,
                'all_scores': all_scores
            })
            
            if predicted == expected:
                correct += 1
                true_positives[expected] += 1
            else:
                false_negatives[expected] += 1
                false_positives[predicted] += 1
        
        # Calculate overall metrics
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-type precision, recall, F1
        relation_types = set(list(true_positives.keys()) + 
                           list(false_positives.keys()) + 
                           list(false_negatives.keys()))
        
        per_type_metrics = {}
        
        for rel_type in relation_types:
            tp = true_positives[rel_type]
            fp = false_positives[rel_type]
            fn = false_negatives[rel_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_type_metrics[rel_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        
        #calculate macro averages
        precisions = [m['precision'] for m in per_type_metrics.values()]
        recalls = [m['recall'] for m in per_type_metrics.values()]
        f1s = [m['f1'] for m in per_type_metrics.values()]
        
        macro_precision = np.mean(precisions) if precisions else 0
        macro_recall = np.mean(recalls) if recalls else 0
        macro_f1 = np.mean(f1s) if f1s else 0
        
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_type': per_type_metrics,
            'predictions': predictions,
            'correct': correct,
            'total': total
        }
        
        # Print summary
        print(f"\nResults:")
        print(f"Accuracy: {accuracy*100:.1f}%")
        print(f"Macro Precision: {macro_precision*100:.1f}%")
        print(f"Macro Recall: {macro_recall*100:.1f}%")
        print(f"Macro F1: {macro_f1*100:.1f}%")
        print(f"\nPer-type metrics:")
        
        for rel_type, metrics in sorted(per_type_metrics.items()):
            print(f"  {rel_type}:")
            print(f"    P: {metrics['precision']*100:.1f}%  "
                  f"R: {metrics['recall']*100:.1f}%  "
                  f"F1: {metrics['f1']*100:.1f}%  "
                  f"(n={metrics['support']})")
        
        return results


# Usage Example
@click.command()
@click.option('--train-json', default='corpus_analysis_results.json', help='Training JSON file path.')
@click.option('--test-json', default='test_dataset.json', help='Test JSON file path.')
@click.option('--use-hypernyms/--no-use-hypernyms', default=True, help='Include hypernyms (r_isa).')
@click.option('--use-trt/--no-use-trt', default=True, help='Include TRT features.')
@click.option('--use-sst/--no-use-sst', default=True, help='Include SST features.')
@click.option('--use-definiteness/--no-use-definiteness', default=False, help='Include definiteness features.')
@click.option('--fusion-threshold', default=0.5, type=float, help='Similarity threshold for merging rules.')
@click.option('--trim-rules/--no-trim-rules', default=False, help='Keep only merged rules.')
def cli(train_json, test_json, use_hypernyms, use_trt, use_sst, use_definiteness, fusion_threshold, trim_rules):
    with open(train_json, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    vector = vectorCreator(
        use_hypernyms=use_hypernyms,
        use_trt=use_trt,
        use_sst=use_sst,
        use_definiteness=use_definiteness,
        fusion_threshold=fusion_threshold,
        trim_rules=trim_rules
    )

    vector.train(train_data)
    results = vector.evaluate(test_data)


    return results

if __name__ == "__main__":
    cli()