# curl http://127.0.0.1:5000/infer
# curl http://172.17.0.2:5000/infer

# curl -X POST -H "Content-Type: application/json" -d '{"datasetName": "test",
# "targetVariable": "var"}' http://127.0.0.1:5000/infer

curl http://172.17.0.2:5000/infer -F file=@assets/data/raw/entrust-highend/simulated_low_end_traces_from_high_end_device.csv
#curl http://127.0.0.1:5000/infer -F file=@assets/data/raw/cnc_without_target/02.csv
# curl http://127.0.0.1:5000/infer -F file=@/media/erikhu/32GB/nova10_downsampled/P2.csv
# curl http://127.0.0.1:5000/infer --data-binary @02.csv -H "Content-Type:text/plain"

# curl http://127.0.0.1:5000/infer
