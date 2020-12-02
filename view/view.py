"""

"""

class View(object):


    def menu(self):
        print(
        "\n"
        "== MENU PRINCIPAL"    
        "\n"
        " 1 - Menu de processamento de imagens: \n 2 - Menu para detecção de objetos \n"
        " 3 - Menu para binarização e correção morfológica com foto \n"
        " 4 - Menu para binarização e correção morfológica com vídeo \n"
        " 5 - SAIR \n"        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções: "))
            if option > 5:
                print("Opção inválida...")
                return self.menu()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.menu()

    
    def submenu(self):
        print(
        "\n"
        "== MENU DE PROCESSAMENTO DE IMAGENS"
        "\n"
        " 1 - Abrir imagem: \n 2 - Recortar imagem: \n 3 - Redimencionar imagem \n"
        " 4 - Tratamento de cores \n 5 - VOLTAR \n"        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções acima: "))
            if option > 5:
                print("Opção inválida...")
                return self.submenu()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.submenu()

    
    def detecmenu(self):
        print(
        "\n"
        "== MENU PARA USO DO ALGORITMO TREINADO E DETECÇÃO DE OBJETOS"
        "\n"
        " 1 - Selecionar uma imagem: \n 2 - Selecionar um vídeo: \n"
        " 3 - Abrir a câmera:  \n 4 - Detectar algum objeto por foto: \n"
        " 5 - Detectar algum objeto pela câmera: \n"
        " 6 - VOLTAR \n"        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções acima: "))
            if option > 6:
                print("Opção inválida...")
                return self.detecmenu()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.detecmenu()


    def sel_algorithm(self):
        print(
        "\n"
        "== MENU PARA ESCOLHA DO ALGORÍTMO "
        "\n"
        " 1 - Cascade - Detectar pessoas \n 2 - Detectar condôminos \n"
        " 3 - VOLTAR "        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções acima: "))
            if option > 3:
                print("Opção inválida...")
                return self.sel_algorithm()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.sel_algorithm()


    def object_color_menu(self):
        print(
        "\n"
        "== MENU PARA DETECÇÃO DE OBJETOS POR CORES DEFINIDAS PELO SISTEMA"
        "\n"
        " 1 - PINK  \n 2 - AMARELO \n 3 - VERDE NEON"
        " 4 - VOLTAR "        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções acima: "))
            if option > 4:
                print("Opção inválida...")
                return self.object_menu()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.object_menu()


    def morphmenu(self, file):
        print(
        "\n"
        "== MENU PARA TRATAMENTO DE BINARIZAÇÃO OU MORFOLÓGICO PARA " + file +
        "\n"
        " 1 - Tratamento com binarização: \n 2 - Tratamento com técnicas de Morfologia \n"
        " 3 - Definir um padrão de cor de objeto com HSV \n 4 - VOLTAR \n"        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções acima: "))
            if option > 4:
                print("Opção inválida...")
                return self.submenu()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.detecmenu()


    def num_iterations(self):        
        try:
            num = int(input("\n Digite aqui o número de interações: "))          
            return num
        except ValueError:
            print("Apenas NÚMEROS!")
            return self.num_iterations()

    
    def select_path(self):
        # example: imagens/imagens_salvas/olhoazul.jpg
        file = input("\n"
        "Digite aqui a URL para a foto ou vídeo que quer usar: "
        )
        return file

    
    def get_parameters(self):
        print("\n"
              " Informe os seguintes parâmetros abaixos para recortar a imagem: ")
        try:
            xh = int(input(" Altura da posição X: "))
            xw = int(input(" Comprimento da posição X: "))
            yh = int(input(" Altura da posição Y: "))
            yw = int(input(" Comprimento da posição Y: "))
            return xh, xw, yh, yw
        except ValueError:
            print("Apenas NÚMEROS!")
            return self.get_parameters()


    def get_wh(self):
        print("\n"
              " Informe a altura e largura que deseja: ")
        try:
            w = int(input(" Largura: "))
            h = int(input(" Altura: "))        
            return w, h
        except ValueError:
            print("Apenas NÚMEROS!")
            return self.get_wh()


    def select_color_option(self):
        print(
        "\n"
        "== MENU PARA ESCOLHER UM PADRÃO DE COR "
        "\n"
        " 1 - GRAY: \n 2 - HSV \n"
        " 3 - LAB \n 4 - VOLTAR \n"        
        "\n"
        )
        try:
            option = int(input("Selecione uma das opções acima: "))
            if option > 4:
                print("Opção inválida...")
                return self.select_color_option()
            else:
                return option

        except ValueError:
            print("Tente apenas uma das opções abaixo!")

            return self.select_color_option()


    def print_message(self):
        print("\n"
            "Não esqueça de anotar os valores de H, S e V"
            "Para fechar a janela precione a tecla ESC."
            "\n"
        )
        
    
    def print_message2(self):
        print("\n"        
        "Para fechar a janela precione a tecla ESC."
        "\n"
        )