# Recommended nodes: 2
# ETA: 7 x 2x4GPUh
# Throughput w/2x4GPU: ~2521
# FLOPs/iter: ~37.613e15
# Total FLOPs: ~94.033e18
source .env
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
	$*
