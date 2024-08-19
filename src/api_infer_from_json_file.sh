# cat assets/data/raw/api_test/nova10.json | curl -X POST -H "Content-Type: application/json" -d @- http://127.0.0.1:5000/infer
# curl -X POST -H "Content-Type: application/json" -d @- http://127.0.0.1:5000/infer
# curl -X POST -H "Content-Type: application/json" -d @assets/data/raw/api_test/nova10.json http://172.17.0.2:5000/infer
curl -X POST -H "Content-Type: application/json" -d @output.json http://172.17.0.2:5000/infer

