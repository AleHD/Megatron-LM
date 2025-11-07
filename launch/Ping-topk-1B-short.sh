# Recommended nodes: 2
source .env
export MBS=1
bash launch/submit.sh 1 \
	--iters 2500 \
	--n-recurrences 5 \
	--n-encode 7 \
	--n-think 4 \
	--n-decode 5 \
	--latent-init truncnorm \
	--think-adapter linear \
	--train-recurrence-method poisson \
	--n-backwards 5 \
	--latent-masker topk \
	--latent-topk-masker-k 128 \
	--linear-adapter-alpha 0.3 \
	$*
