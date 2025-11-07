source .env
bash launch/submit.sh 390 \
	--scratch \
	--opt muon \
	--iters 25000 \
	--n-recurrences 5 \
	--n-encode 7 \
	--n-think 4 \
	--n-decode 5 \
	--latent-init truncnorm \
	--think-adapter linear \
	--train-recurrence-method poisson \
	--n-backwards 5 \
	$*
