for data in dl19 dl20;
    do for topk in 20 100;
        do for model in output_v2_aug_vicuna_7b output_v2_vicuna_7b;
            do for fstage in BM25 SPLADE_P_P_ENSEMBLE_DISTIL;
                do echo $data $topk $model $fstage;
                ls -l rerank_results/$fstage/*$model*$topk*$data*shuffled*T* | wc -l;
                for file in rerank_results/$fstage/*$model*$topk*$data*shuffled*T*;
                    do # if data is dl19
                        if [ $data = "dl19" ]; then
                             $TEVAL_MSP_DL19 $file >> logs/$fstage.$model.$topk.$data;
                        else
                             $TEVAL_MSP_DL20 $file >> logs/$fstage.$model.$topk.$data;
                        fi;
                    done;
            done;
        done;
    done;
done;



for data in dl19 dl20;
    do for topk in 20 100;
        do for model in output_v2_aug_vicuna_7b output_v2_vicuna_7b;
            do for fstage in BM25 SPLADE_P_P_ENSEMBLE_DISTIL;
                do echo $data $topk $model $fstage;
                # Do not match shuffled
                ls -l rerank_results/$fstage/*$model*$topk*$data*T* | grep -v shuffled | wc -l;
                # For all file not with shuffled
                for file in rerank_results/$fstage/*$model*$topk*$data*T*;
                    do # if data is dl19
                        # if shuffled in file skipe
                        if [[ $file == *"shuffled"* ]]; then
                            continue;
                        fi;
                        echo $file;
                        if [ $data = "dl19" ]; then
                             $TEVAL_MSP_DL19 $file >> logs/ns.$fstage.$model.$topk.$data;
                        else
                             $TEVAL_MSP_DL20 $file >> logs/ns.$fstage.$model.$topk.$data;
                        fi;
                    done;
            done;
        done;
    done;
done;
        
for data in dl19 dl20;
    do for topk in 20 100;
        do for model in output_v2_aug_vicuna_7b;
            do for fstage in BM25_RM3 D_BERT_KD_TASB OPEN_AI_ADA2;
                do echo $data $topk $model $fstage;
                # Do not match shuffled
                ls -l rerank_results/$fstage/*$model*$topk*$data*T* | grep -v shuffled | wc -l;
                # For all file not with shuffled
                for file in rerank_results/$fstage/*$model*$topk*$data*T*;
                    do # if data is dl19
                        # if shuffled in file skipe
                        if [[ $file == *"shuffled"* ]]; then
                            continue;
                        fi;
                        echo $file;
                        if [ $data = "dl19" ]; then
                             $TEVAL_MSP_DL19 $file >> logs/ns.$fstage.$model.$topk.$data;
                        else
                             $TEVAL_MSP_DL20 $file >> logs/ns.$fstage.$model.$topk.$data;
                        fi;
                    done;
            done;
        done;
    done;
done;


# All fstages
for fstage in BM25  BM25_RM3  D_BERT_KD_TASB  OPEN_AI_ADA2  SPLADE_P_P_ENSEMBLE_DISTIL;
    do echo $fstage;
    for data in dl19 dl20;
        do echo $data $fstage
        if [ $data = "dl19" ]; then
            $TEVAL_MSP_DL19 retrieve_results/$fstage/*$data.txt >> logs/fs.$fstage.$data;
        else
            $TEVAL_MSP_DL20 retrieve_results/$fstage/*$data.txt >> logs/fs.$fstage.$data;
        fi;
        done;
    done;


for fstage in BM25  BM25_RM3  OPEN_AI_ADA2 D_BERT_KD_TASB SPLADE_P_P_ENSEMBLE_DISTIL;
    do echo $fstage
    for file in logs/fs.$fstage.*;
        do ndcg=$(egrep ndcg_cut_10 $file | cut -f3); map=$(egrep map_cut_100 $file | cut -f3); echo -n \\multicolumn{2}{l}{$ndcg}{$map} \& " "; done;
        echo ""; done;


for fstage in BM25  BM25_RM3 OPEN_AI_ADA2 D_BERT_KD_TASB SPLADE_P_P_ENSEMBLE_DISTIL;
    do echo $fstage
    for topk in 20 100;
    do for model in output_v2_aug_vicuna_7b output_v2_vicuna_7b;
        do for dataset in dl19 dl20;
        do for file in logs/ns.$fstage.*$model*$topk*$dataset;
        do ndcg=$(egrep ndcg_cut_10 $file | cut -f3); map=$(egrep map_cut_100 $file | cut -f3); 
        if [ $dataset = "dl19" ]; then 
            echo "";
            echo $file; echo -n \\multicolumn{2}{l}{$ndcg}{$map} \& " "; 
        else 
            echo -n \\multicolumn{2}{l}{$ndcg}{$map};
        fi; done;
     done; done; done; done;

for fstage in BM25 SPLADE_P_P_ENSEMBLE_DISTIL;
    do echo $fstage
    for file in logs/$fstage.*;
        do echo $file; ndcg=$(egrep ndcg_cut_10 $file | cut -f3); map=$(egrep map_cut_100 $file | cut -f3); echo $ndcg \& $map ; done;
        echo ""; done;