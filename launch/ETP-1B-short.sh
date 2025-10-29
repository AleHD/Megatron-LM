source .env
bash launch/submit.sh 1 \
	--iters 2500 \
	--n-recurrences 5 \
	--n-encode 7 \
	--n-think 4 \
	--n-decode 5 \
	$1
