#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]
then
  echo "Usage: $0 <out_dir> <text_level> <in_dir> <flags>" 2>&1
  exit 1
fi
out_dir="$1"
text_level="$2"
in_dir="$3"
shift;
shift;
shift;
flags="$@"
time python3 -m NAACL.backoffnet --ds-train-dev-file ${in_dir}/${text_level}/ds_train_dev.txt --jax-dev-test-file ${in_dir}/${text_level}/jax_dev_test.txt --text-level ${text_level} -o ${out_dir} --try-all-checkpoints ${flags}
dev_preds=`ls ${out_dir}/pred_dev*.tsv`
test_preds=`ls ${out_dir}/pred_test*.tsv`

for fn in ${dev_preds} ${test_preds}
do
  echo "$fn"
  python3 -m NAACL.prune_pred_gv_map entity_tokens_eval/entities0.tsv entity_tokens_eval/tokens.json ${fn} ${fn}.pruned
#  python3 -m machinereading.evaluation.eval ${fn}
done

for fn in ${dev_preds}
do
  echo "$fn"
#  python3 -m machinereading.evaluation.eval ${fn}.pruned
done

for fn in ${test_preds}
do
  echo "$fn"
#  python3 -m machinereading.evaluation.eval --test ${fn}.pruned
done