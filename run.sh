echo "Iniciando container..."
docker run -d \
    --name rag-app-i0 \
    --env-file .env.prod \
    --network rag-app-network \
    -p 8501:8501 \
    -p 5000:5000 \
    rag-app
echo "Container iniciado com sucesso!"
echo "Vá para http://localhost:8501 para interagir com o Mestre dos PDFs"
echo "Vá para http://localhost:5000/docs para visualizar o Swagger das rotas da API"