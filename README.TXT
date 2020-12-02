""""""""Bem vindo ao programa de reconhecimento de moradores de condomínios!!!!

"== Estrutura do projeto "


-> FINAL

--> controller
----> controller.py

--> imagens
----> condominos
----> imagens_de_log
----> imagens_salvas
--------> imagens_morfo
----> negativas
----> positivas
----> treinamento
----> videos_salvos

--> services
----> services.py

--> utils
----> renamefiles.py
----> buildListNegative.py

--> view
----> view.py

exec.py
README.TXT


---------------------------------------------------------------------

===== DETALHES DOS ARQUIVOS:


* controller/controller.py:
- Camada do sistema responsável por coletar as informações da view e transformar em requisições 
para a camada de serviço.

* Pasta de imagens:
- Diretório onde todas as imagens do sistemas são armazenadas;
Pasta condominos = Imagens dos moradores do prédio - usadas para o treino do face_recognition;
Pasta imagens_de_log = Imagens salvas pelas funções que detectam movimentos ou objetos;
Pasta imagens_salvas = Imagens para serem usadas em exemplos e diretório onde as imagens recortadas e 
pré-processadas são armazenadas;
Pasta imagens_morfo = Imagens que são salvas após tratamento morfológico; 
Pasta negativas = Imagens negativas para o treinamento do algoritmo Cascade do OpenCV;
Pasta positivas = Imagens positivas para o treinamento do algoritmo Cascade do OpenCV;
Pasta treinamento = Pasta onde os arquivos do treinamento são salvas. 
Pasta videos_salvos = Pastas onde os vídeos utilizados no sistema são armazenados;

* services/services.py:
- Camada de serviço da aplicação onde todas as funções de processamento e treinamento de imagens estão.

* utils/buildListNegative.py:
- Side function para uso em treinamento do algoritmo Cascade do OpenCV.

* utils/renamefiles.py:
- Side function para uso em treinamento do algoritmo Cascade do OpenCV.

* view/view.py:
- Camada da aplicação que interage com o usuário para obter os inputs. 

* exec.py:
- Comando de iniciação da aplicação. 

* README.TXT:
- Arquivo com a documentação do sistema.


---------------------------------------------------------------------

===== TREINAMENTO DO ALGORITMO:


* Download e configuração do OpenCV:

A próxima etapa é fazer o download da biblioteca OpenCV para treinar um modelo de aprendizado de máquina para usar neste projeto.
Baixe o OpenCV em https://opencv.org/, selecione a opção releases e escolha a versão 3.4.12 (recomendado para python versão 3.6.1)
Depois de fazer isso, o download está prestes a começar. Quando o download for concluído, descompacte a pasta OpenCV baixada e salve em outro diretório.

Para usar as funções OpenCV, salve o seguinte caminho nas variáveis ​​de ambiente:
C: \ Usuários \ {seu usuário} \ {sua pasta} \ opencv \ build \ x64 \ vc14 \ bin

Agora você pode usar todas as funções do OpenCV localizadas dentro da pasta bin.
Este projeto usa as seguintes funções: opencv_annotation, opencv_createsamples e opencv_traincascade.

* Execução de funções paralelas:

Neste projeto, você encontrará funções secundárias chamadas buildListNegative.py e renamefiles.py.
buildListNegative.py precisa ser executado para criar um arquivo de texto com todas as imagens negativas. Este arquivo você usa no processo de algoritmo de treinamento.
renamefiles.py é uma função secundária que nos ajuda a renomear todas as imagens nas pastas de negativos e positivos.

* Execute opencv_annotation:

Abra o Prompt de Comando em seu sistema operacional e navegue até as pastas de imagens.
dentro da pasta selecionada, anote a instrução: opencv_annotation --annotations = {seu nome de arquivo de saída} .txt --images = {as pastas de imagens positivas} /

A próxima etapa é identificar o objeto que você deseja detectar para treinar o modelo. Para fazer isso, basta seguir estas etapas simples:
- Desenhado um retângulo ao redor do objeto.
- Pressione C para confirmar.
- Pressione N para ir para a próxima imagem.

O resultado será salvo no arquivo de saída com todas as coordenadas de todas as imagens.

* Execute opencv_createsamples:

dentro da mesma pasta selecionada, anote as instruções:
opencv_createsamples -info {seu nome de arquivo de saída} .txt -bg {seu nome de arquivo negativo} .txt -vec {seu nome de arquivo vetorial} .vec -w {largura} -h {altura}
EX: opencv_createsamples -info exit.txt -bg negatives.txt -vec vetor.vec -w 24 -h 24

* Execute opencv_traincascade:

dentro da mesma pasta selecionada, anote as instruções:
opencv_traincascade -data {seu nome de arquivo de saída} -vec {seu nome de arquivo vetorial} .vec -bg {seu nome de arquivo de negativos} .txt
-numPos {número de imagens positivas} -numNeg {número de imagens negativas} -w {largura} -h {altura} -precalcValBufSize 1024 -precalcIdxBufSize 1024
-numStages {número de estágios de treinamento} -acceptanceRatioBreakValue {valor de aceitação da aprendizagem}
EX: opencv_traincascade -data training -vec vector.vec -bg negatives.txt -numPos 100 -numNeg 200 -w 24 -h 24 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -numStages 30 -acceptanceRatioBreakValue 1.0e-5


---------------------------------------------------------------------

===== EXECUÇÃO:

Inicialmente, para o funcionamento da aplicação, é necessária a instalação das seguintes bibliotecas:
-> OpenCV
-> Matplotlib
-> Numpy 
-> Dlib 
-> Cmake
-> face_recognition

OBS: Para o bom funcionamento da aplicação, certifique-se de baixar a versão 3.6.2 do python.

Para a execução da aplicação, apenas execute a instrução: python exec.py em seu terminar. Se tudo 
estiver em ordem, o menu PRINCIPAL deverá aparecer em sua tela.


---------------------------------------------------------------------

"== MENU PRINCIPAL"    
        
        1 - Menu de processamento de imagens: 
        2 - Menu para detecção de objetos 
        3 - Menu para binarização e correção morfológica com foto 
        4 - Menu para binarização e correção morfológica com vídeo 
        5 - SAIR     

        {

            Menu com as funções principais do aplicação. 
            Opção 1 = processamento das imagens - recorte, troca de cores, redimencionamento...
            Opção 2 = Opção para detectar objetos e pessoas.
            Opção 3 = Tratamento morfológico de fotos e binarização.
            Opção 4 = Tratamento morfológico de vídeos e binarização.
            Opção 5 = Sair do sistema.

        }



"== MENU DE PROCESSAMENTO DE IMAGENS"
        
        1 - Abrir imagem: 
        2 - Recortar imagem: 
        3 - Redimencionar imagem 
        4 - Tratamento de cores  
        5 - VOLTAR 


        {

            Menu com as funções de processamento de imagem.
            Opção 1 = Simples visualização.
            Opção 2 = Realizar o recorte de imagem.
            Opção 3 = Redimencionar uma imagem.
            Opção 4 = Aplicar um tratamento de cores na imagem.
            Opção 5 = Voltar
            
        }



"== MENU PARA USO DO ALGORITMO TREINADO E DETECÇÃO DE OBJETOS"
        
        1 - Selecionar uma imagem: 
        2 - Selecionar um vídeo: 
        3 - Abrir a câmera:  
        4 - Detectar algum objeto por foto: 
        5 - Detectar algum objeto pela câmera: 
        6 - VOLTAR 


        {

            Menu para escolher as diferentes formas de detectar objetos e pessoas no condomínio.
            Opção 1 = Escolher uma imagem de uma pessoa.
            Opção 2 = Escolher um vídeo de uma pessoa.
            Opção 3 = Detectar pessoas através de uma camera.
            Opção 4 = Função para detectar algum objeto por foto.
            Opção 5 = Função para detectar algum objeto através da camera.
            Opção 6 = Sair.

        }



"== MENU PARA ESCOLHA DO ALGORÍTMO "
        
        1 - Cascade - Detectar pessoas 
        2 - Detectar condôminos 
        3 - VOLTAR    


        {

            Menu para escolha dos algoritmos de inteligência. 
            Opção 1 = Utilizar a função Cascade do OpenCV para detectar pessoas.
            Opção 2 = Utilizar o face_recognition para detectar moradores. 
            Opção 3 = Voltar. 

        }         
         


"== MENU PARA DETECÇÃO DE OBJETOS POR CORES DEFINIDAS PELO SISTEMA"
        
        1 - PINK  
        2 - AMARELO 
        3 - VERDE NEON
        4 - VOLTAR    


        {

            Menu para escolher as cores dos objetos a serem detectados a partir de opções definidas.
            Opção 1 = Objeto de cor rosa.
            Opção 2 = Objeto de cor amarela.
            Opção 3 = Objeto de cor verde claro.
            Opção 4 = Voltar.

        }
        


"== MENU PARA TRATAMENTO DE BINARIZAÇÃO OU MORFOLÓGICO PARA " 
       
        1 - Tratamento com binarização: 
        2 - Tratamento com técnicas de Morfologia 
        3 - Definir um padrão de cor de objeto com HSV 
        4 - VOLTAR         


        {

            Menu para tratamento avançado de imagens.
            Opção 1 = Tratar uma imagem aplicando diversas técnicas de binarização. 
            Opção 2 = Tratar uma imagem aplicando diversas técnicas de morfologia. 
            Opção 3 = Utilizar a função para definir uma nova cor a partir do HSV.
            Opção 4 = Voltar.

        }
        


"== MENU PARA ESCOLHER UM PADRÃO DE COR "
        
        1 - GRAY 
        2 - HSV 
        3 - LAB 
        4 - VOLTAR   


        {

            Menu para escolha de padrão de confirmar.
            Opção 1 = Aplicar na imagem uma escala em cinza.
            Opção 2 = Aplicar na imagem uma escala de cores em padrão HSV.
            Opção 3 = Aplicar na imagem uma escala de cores em padrão LAB.
            Opção 4 = Voltar.

        }  


