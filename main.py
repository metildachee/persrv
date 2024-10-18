import pandas as pd
import csv
import os
import time
from metrics import calculate_mmrr, calculate_sum_hits_percentage, calculate_mrr
from datetime import datetime
import argparse

from prediction_fn import predict_by_paddleocr_rawqueries_llava_desc_emotions_finetuned_keywords_versatility
def run_experiment(experiment_title, test_file, output_dir, similarity_file, counts, hyp1, hyp2, hyp3):
    df = pd.read_csv(test_file, header=None, names=['uid', 'query', 'docid'])
    df.sort_values(by=['uid', 'query'], inplace=True)
    user_query_groups = df.groupby(['uid', 'query'])['docid'].apply(list).to_dict()
    user_mrr = {}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting {experiment_title}, hyperparameters: {counts}, {hyp2}, {hyp1}, {hyp3}...")

    total_queries = 0
    total_time = 0
    for count in counts:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_hits_csv = os.path.join(output_dir, f"log_{experiment_title}_{count}_{current_time}.csv")
        if not os.path.exists(output_hits_csv):
            with open(output_hits_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["userid", "query", "ground_truths", "predictions", f"multi_mrr@{count}", f"recall@{count}"])

        user_mrr = {}
        for (user_id, query), docids in user_query_groups.items():
            if user_id == "uid":
                continue

            start_time = time.time()
            hits = predict_by_paddleocr_rawqueries_llava_desc_emotions_finetuned_keywords_versatility(query, user_id, emotions_boost=hyp1, finetune_keyword=hyp2, desc_boost=hyp3, size=count, userid_centroids_file_path=similarity_file)
            elapsed_time = time.time() - start_time

            hits = [hit['id'] for hit in hits]

            if user_id not in user_mrr:
                user_mrr[user_id] = {}

            if query not in user_mrr[user_id]:
                user_mrr[user_id][query] = {'predictions': [], 'ground_truths': []}

            total_time += elapsed_time
            total_queries += 1

            user_mrr[user_id][query]['predictions'].extend(hits)
            user_mrr[user_id][query]['ground_truths'].extend(docids)

            mrr = calculate_mrr([user_mrr[user_id][query]['predictions']], [user_mrr[user_id][query]['ground_truths']])
            multi_mrr = calculate_mmrr([user_mrr[user_id][query]['predictions']], [user_mrr[user_id][query]['ground_truths']])
            top_n_sumhits, _ = calculate_sum_hits_percentage([user_mrr[user_id][query]['predictions']], [user_mrr[user_id][query]['ground_truths']])

            with open(output_hits_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([user_id, query, docids, hits, multi_mrr, top_n_sumhits])

            print(f"Written to: {output_hits_csv}")

        hyper_desc = f'Hyperparameters: count={count}, hyp1={hyp1}, hyp2={hyp2}, hyp3={hyp3}'
        print(hyper_desc)

        total_mrr = 0
        total_hit_rate = 0
        num_users = len(user_mrr)

        for user_id, data in user_mrr.items():
            mrr = calculate_mrr([d['predictions'] for d in data.values()], [d['ground_truths'] for d in data.values()])
            multi_mrr = calculate_mmrr([d['predictions'] for d in data.values()], [d['ground_truths'] for d in data.values()])
            top_n_sumhits, ranks = calculate_sum_hits_percentage([d['predictions'] for d in data.values()], [d['ground_truths'] for d in data.values()])

            print(f"User {user_id} MRR:", mrr)
            print(f"User {user_id} Multi-MRR:", multi_mrr)
            print(f"User {user_id} Percentage of hits found in ground truth ({count}): {top_n_sumhits}%")

            total_mrr += multi_mrr
            total_hit_rate += top_n_sumhits

        avg_mrr = total_mrr / num_users if num_users > 0 else 0
        avg_hit_rate = total_hit_rate / num_users if num_users > 0 else 0

        print(f"[{experiment_title}] Average Multi-MRR for {hyper_desc}: {avg_mrr}")
        print(f"[{experiment_title}] Average Percentage of hits found in ground truth ({count}): {avg_hit_rate}%")
        print()

    avg_time = total_time / total_queries if total_queries > 0 else 0
    print(f"Average time taken for user-query pairs: {avg_time:.4f} seconds, {total_time} total_time {total_queries} queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run an experiment.')
    parser.add_argument('--experiment', type=str, required=True, help='The name of the experiment to run.')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test file (CSV format).')
    parser.add_argument('--similarity_file', type=str, required=True, help='Path to the userid-centroids (CSV format).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--counts', nargs='+', type=int, default=[10, 20, 50, 100], help='List of counts for evaluation (default: [10, 20, 50, 100]).')
    parser.add_argument('--hyp1', type=float, default=0.3, help='Hyperparameter 1 for emotions_boost (default: 0.3).')
    parser.add_argument('--hyp2', type=float, default=0, help='Hyperparameter 2 for finetune_keyword (default: 0).')
    parser.add_argument('--hyp3', type=float, default=1.0, help='Hyperparameter 3 for desc_boost (default: 1.0).')
    
    args = parser.parse_args()
    
    run_experiment(experiment_title=args.experiment, 
                   test_file=args.test_file, 
                   output_dir=args.output_dir, 
                   similarity_file=args.similarity_file,
                   counts=args.counts, 
                   hyp1=args.hyp1, 
                   hyp2=args.hyp2, 
                   hyp3=args.hyp3)
