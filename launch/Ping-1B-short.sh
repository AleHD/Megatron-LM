# Recommended nodes: 3
# ETA: ~12 x 1x4GPUh
# Throughput w/4GPU: 9080
# FLOPs/iter: 109.307e15
# Total FLOPs: 273.267
source .env
bash launch/submit.sh 1 \
	--iters 2500 \
	--n-encode 4 \
	--n-think 8 \
	--n-decode 4 \
	--n-recurrences 16 \
	--latent-init truncnorm \
	--think-adapter linear \
	--train-recurrence-method poisson \
	--n-backwards 8 \
	$*
