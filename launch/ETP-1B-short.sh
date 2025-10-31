# Recommended nodes: 2
# ETA: ~8 x 2x4GPUh
# Throughput w/2x4GPU: ~2331
# FLOPs/iter: ~37.405e15
# Total FLOPs: ~93.513e18
source .env
bash launch/submit.sh 1 \
	--iters 2500 \
	--n-recurrences 5 \
	--n-encode 7 \
	--n-think 4 \
	--n-decode 5 \
	$*
