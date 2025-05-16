echo "Iniciando Attu UI..."
docker run -p 3000:3000 -d --name attu-ui --network rag-app-network -e MILVUS_URL=192.168.0.9:19530 zilliz/attu:v2.5
echo "Attu UI iniciado com sucesso. Acesse em http://localhost:3000"