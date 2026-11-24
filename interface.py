
import os
import re
import cv2
import numpy as np
import customtkinter as ctk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "detector_valores-3", "weights", "best.pt"
)

VALORES_MOEDAS = (5, 10, 25, 50, 100)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# APLICAÇÃO
# ============================================================

class Aplicativo(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Detector de Moedas")
        self.geometry("1100x720")
        self.minsize(900, 600)

        # Dados da imagem e das detecções
        self.caminho_imagem = None
        self.imagem_original = None
        self.resultado_yolo = None
        self.moedas_associadas = []
        self.filtro_atual = None
        self.imagem_tk = None

        # Carrega o modelo
        if not os.path.isfile(MODELO_PATH):
            messagebox.showerror(
                "Erro",
                "Modelo não encontrado!\n\n"
                f"Verifique o caminho:\n{MODELO_PATH}"
            )
            self.destroy()
            return

        try:
            self.modelo = YOLO(MODELO_PATH)
        except Exception as erro:
            messagebox.showerror(
                "Erro ao carregar o modelo",
                str(erro)
            )
            self.destroy()
            return

        # Monta a interface
        self.construir_interface()

    # ========================================================
    # INTERFACE PRINCIPAL
    # ========================================================

    def construir_interface(self):

        # Faixa superior
        self.faixa_superior = ctk.CTkFrame(
            self,
            height=8,
            corner_radius=0,
            fg_color="#D4A017"
        )
        self.faixa_superior.pack(
            side="top",
            fill="x"
        )

        # Área principal
        self.conteudo = ctk.CTkFrame(
            self,
            corner_radius=0
        )
        self.conteudo.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12
        )

        self.conteudo.grid_columnconfigure(0, weight=0)
        self.conteudo.grid_columnconfigure(1, weight=1)
        self.conteudo.grid_rowconfigure(0, weight=1)

        # Painel lateral
        self.painel_lateral = ctk.CTkFrame(
            self.conteudo,
            width=240
        )
        self.painel_lateral.grid(
            row=0,
            column=0,
            sticky="ns",
            padx=(0, 12),
            pady=0
        )
        self.painel_lateral.grid_propagate(False)

        titulo = ctk.CTkLabel(
            self.painel_lateral,
            text="DETECTOR\nDE MOEDAS",
            font=("Arial", 24, "bold"),
            text_color="#D4A017"
        )
        titulo.pack(
            padx=12,
            pady=(25, 20)
        )

        # Botão para selecionar imagem
        self.botao_imagem = ctk.CTkButton(
            self.painel_lateral,
            text="Selecionar Imagem",
            command=self.inserir_imagem,
            height=40
        )
        self.botao_imagem.pack(
            padx=15,
            pady=10,
            fill="x"
        )

        # Botão para executar detecção
        self.botao_detectar = ctk.CTkButton(
            self.painel_lateral,
            text="Detectar Moedas",
            command=self.detectar_moedas,
            height=40,
            fg_color="#228B22",
            hover_color="#176B17"
        )
        self.botao_detectar.pack(
            padx=15,
            pady=10,
            fill="x"
        )

        # Prévia da imagem
        self.titulo_previa = ctk.CTkLabel(
            self.painel_lateral,
            text="Imagem selecionada",
            font=("Arial", 14, "bold")
        )
        self.titulo_previa.pack(
            padx=10,
            pady=(20, 5)
        )

        self.previa = ctk.CTkLabel(
            self.painel_lateral,
            text="Nenhuma imagem\nselecionada",
            width=200,
            height=150
        )
        self.previa.pack(
            padx=10,
            pady=5
        )

        # Alternância entre tema claro e escuro
        self.interruptor_tema = ctk.CTkSwitch(
            self.painel_lateral,
            text="Modo escuro",
            command=self.alternar_tema
        )
        self.interruptor_tema.select()
        self.interruptor_tema.pack(
            padx=15,
            pady=(25, 10)
        )

        # Abas
        self.abas = ctk.CTkTabview(
            self.conteudo
        )
        self.abas.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.aba_principal = self.abas.add("Principal")
        self.abafiltro = self.abas.add("Filtro")

        self.construir_aba_principal()
        self.construir_aba_filtro()

        # Faixa inferior
        self.faixa_inferior = ctk.CTkFrame(
            self,
            height=8,
            corner_radius=0,
            fg_color="#D4A017"
        )
        self.faixa_inferior.pack(
            side="bottom",
            fill="x"
        )

    # ========================================================
    # ABA PRINCIPAL
    # ========================================================

    def construir_aba_principal(self):

        self.aba_principal.grid_columnconfigure(
            0,
            weight=1
        )
        self.aba_principal.grid_rowconfigure(
            1,
            weight=1
        )

        self.titulo_resultado = ctk.CTkLabel(
            self.aba_principal,
            text="Resultado da Detecção",
            font=("Arial", 20, "bold")
        )
        self.titulo_resultado.grid(
            row=0,
            column=0,
            padx=10,
            pady=15
        )

        self.area_resultado = ctk.CTkLabel(
            self.aba_principal,
            text="Selecione uma imagem e clique em Detectar Moedas.",
            font=("Arial", 15)
        )
        self.area_resultado.grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        self.status = ctk.CTkLabel(
            self.aba_principal,
            text="Aguardando imagem...",
            font=("Arial", 14)
        )
        self.status.grid(
            row=2,
            column=0,
            padx=10,
            pady=15
        )

    # ========================================================
    # ABA DE FILTROS
    # ========================================================

    def construir_aba_filtro(self):

        for coluna in range(3):
            self.abafiltro.grid_columnconfigure(
                coluna,
                weight=1
            )

        self.status_filtro = ctk.CTkLabel(
            self.abafiltro,
            text="Selecione o valor da moeda",
            font=("Arial", 18, "bold")
        )
        self.status_filtro.grid(
            row=0,
            column=0,
            columnspan=3,
            padx=10,
            pady=20
        )

        # Filtros de 5, 10, 25, 50 e 100 centavos
        valores = (5, 10, 25, 50, 100)

        for indice, centavos in enumerate(valores):

            linha = 1 + indice // 3
            coluna = indice % 3

            botao = ctk.CTkButton(
                self.abafiltro,
                text=f"{centavos} Centavos",
                command=lambda valor=centavos:
                    self.selecionar_filtro(valor)
            )

            botao.grid(
                row=linha,
                column=coluna,
                sticky="ew",
                padx=8,
                pady=10
            )

        # Botão para remover o filtro
        self.botao_todas = ctk.CTkButton(
            self.abafiltro,
            text="Mostrar Todas",
            command=self.limpar_filtro
        )

        self.botao_todas.grid(
            row=3,
            column=0,
            columnspan=3,
            padx=10,
            pady=20,
            sticky="ew"
        )

        self.resultado_filtro = ctk.CTkLabel(
            self.abafiltro,
            text="Faça uma detecção para utilizar os filtros.",
            font=("Arial", 14)
        )
        self.resultado_filtro.grid(
            row=4,
            column=0,
            columnspan=3,
            padx=10,
            pady=10
        )

    # ========================================================
    # NORMALIZAÇÃO DOS NOMES DAS CLASSES
    # ========================================================

    def normalizar_nome(self, nome):

        return re.sub(
            r"[^a-z0-9]",
            "",
            str(nome).lower()
        )

    # ========================================================
    # SELECIONAR IMAGEM
    # ========================================================

    def inserir_imagem(self):

        caminho = filedialog.askopenfilename(
            title="Selecionar imagem de moedas",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if not caminho:
            return

        imagem = cv2.imread(caminho)

        if imagem is None:
            messagebox.showerror(
                "Erro",
                "Não foi possível abrir a imagem selecionada."
            )
            return

        self.caminho_imagem = caminho
        self.imagem_original = imagem

        # Limpa os resultados da imagem anterior
        self.resultado_yolo = None
        self.moedas_associadas = []
        self.filtro_atual = None

        self.exibir_imagem(
            imagem,
            self.previa,
            tamanho=(200, 150)
        )

        self.area_resultado.configure(
            image=None,
            text="Clique em Detectar Moedas."
        )

        self.status.configure(
            text="Imagem carregada com sucesso."
        )

        self.resultado_filtro.configure(
            text="Faça uma detecção para utilizar os filtros.",
            image=None
        )

        self.status_filtro.configure(
            text="Selecione o valor da moeda"
        )

    # ========================================================
    # EXIBIR IMAGEM NA INTERFACE
    # ========================================================

    def exibir_imagem(
        self,
        imagem,
        componente,
        tamanho=(650, 450)
    ):

        if imagem is None:
            return

        # OpenCV BGR para RGB
        imagem_rgb = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )

        imagem_pil = Image.fromarray(imagem_rgb)

        # Redimensiona sem distorcer
        imagem_pil.thumbnail(
            tamanho,
            Image.Resampling.LANCZOS
        )

        imagem_ctk = ctk.CTkImage(
            light_image=imagem_pil,
            dark_image=imagem_pil,
            size=imagem_pil.size
        )

        componente.configure(
            image=imagem_ctk,
            text=""
        )

        # Mantém a referência da imagem
        componente.imagem_referencia = imagem_ctk

    # ========================================================
    # DETECTAR MOEDAS COM YOLO
    # ========================================================

    def detectar_moedas(self):

        if self.imagem_original is None:
            messagebox.showwarning(
                "Atenção",
                "Selecione uma imagem primeiro."
            )
            return

        try:
            self.status.configure(
                text="Detectando moedas..."
            )
            self.update_idletasks()

            resultados = self.modelo.predict(
                source=self.imagem_original,
                conf=0.25,
                verbose=False
            )

            resultado = resultados[0]

            # Resultado anotado pelo YOLO
            self.resultado_yolo = resultado.plot()

            # Recupera as caixas detectadas
            nomes = self.modelo.names
            deteccoes = []

            for caixa in resultado.boxes:

                classe_id = int(caixa.cls[0].item())

                if isinstance(nomes, dict):
                    nome_classe = nomes[classe_id]
                else:
                    nome_classe = nomes[classe_id]

                x1, y1, x2, y2 = (
                    int(valor)
                    for valor in caixa.xyxy[0].tolist()
                )

                confianca = float(caixa.conf[0].item())

                deteccoes.append({
                    "nome": str(nome_classe),
                    "nome_normalizado": self.normalizar_nome(
                        nome_classe
                    ),
                    "caixa": (x1, y1, x2, y2),
                    "confianca": confianca
                })

            # Identifica as caixas de moedas
            caixas_moedas = [
                d for d in deteccoes
                if d["nome_normalizado"] == "moeda"
            ]

            # Identifica as caixas de valores
            classes_valores = {
                self.normalizar_nome(f"{valor} Centavos"): valor
                for valor in VALORES_MOEDAS
            }

            deteccoes_valores = [
                d for d in deteccoes
                if d["nome_normalizado"] in classes_valores
            ]

            self.moedas_associadas = []

            # Associa cada valor à caixa da moeda correspondente
            for moeda in caixas_moedas:

                x1, y1, x2, y2 = moeda["caixa"]

                melhor_valor = None
                melhor_confianca = -1

                for deteccao in deteccoes_valores:

                    vx1, vy1, vx2, vy2 = deteccao["caixa"]

                    centro_x = (vx1 + vx2) / 2
                    centro_y = (vy1 + vy2) / 2

                    # O centro da caixa de valor deve estar
                    # dentro da caixa da moeda
                    dentro_da_moeda = (
                        x1 <= centro_x <= x2
                        and y1 <= centro_y <= y2
                    )

                    if dentro_da_moeda:
                        if deteccao["confianca"] > melhor_confianca:

                            melhor_confianca = deteccao["confianca"]

                            melhor_valor = deteccao["nome"]

                self.moedas_associadas.append({
                    "caixa": moeda["caixa"],
                    "valor": melhor_valor
                })

            self.filtro_atual = None

            # Mostra todas as detecções do YOLO
            self.exibir_imagem(
                self.resultado_yolo,
                self.area_resultado
            )

            quantidade_total = len(self.moedas_associadas)

            # Conta moedas associadas a cada valor
            contagem = {
                valor: sum(
                    1 for moeda in self.moedas_associadas
                    if moeda["valor"] is not None
                    and self.normalizar_nome(moeda["valor"])
                    == self.normalizar_nome(f"{valor} Centavos")
                )
                for valor in VALORES_MOEDAS
            }

            texto_contagem = " | ".join(
                f"{valor}¢: {contagem[valor]}"
                for valor in VALORES_MOEDAS
            )

            self.status.configure(
                text=(
                    f"Moedas detectadas: {quantidade_total}\n"
                    f"{texto_contagem}"
                )
            )

            self.status_filtro.configure(
                text="Escolha um filtro para visualizar as moedas"
            )

            self.resultado_filtro.configure(
                text="Selecione um dos valores acima."
            )

        except Exception as erro:
            messagebox.showerror(
                "Erro na detecção",
                f"Não foi possível detectar as moedas:\n\n{erro}"
            )

            self.status.configure(
                text="Ocorreu um erro na detecção."
            )

    # ========================================================
    # APLICAR FILTRO
    # ========================================================

    def selecionar_filtro(self, valor):

        if self.imagem_original is None:
            messagebox.showwarning(
                "Atenção",
                "Selecione uma imagem primeiro."
            )
            return

        if self.resultado_yolo is None:
            messagebox.showwarning(
                "Atenção",
                "Clique em Detectar Moedas antes de aplicar o filtro."
            )
            return

        self.filtro_atual = valor

        imagem_filtrada = self.imagem_original.copy()

        filtro_normalizado = self.normalizar_nome(
            f"{valor} Centavos"
        )

        moedas_filtradas = [
            moeda for moeda in self.moedas_associadas
            if moeda["valor"] is not None
            and self.normalizar_nome(moeda["valor"])
            == filtro_normalizado
        ]

        # Desenha somente as caixas das moedas selecionadas
        for moeda in moedas_filtradas:

            x1, y1, x2, y2 = moeda["caixa"]

            cv2.rectangle(
                imagem_filtrada,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

        self.exibir_imagem(
            imagem_filtrada,
            self.resultado_filtro
        )

        self.status_filtro.configure(
            text=f"Filtro selecionado: {valor} Centavos"
        )

        self.abas.set("Filtro")

        quantidade = len(moedas_filtradas)

        self.status.configure(
            text=f"Moedas de {valor} centavos encontradas: {quantidade}"
        )

    # ========================================================
    # MOSTRAR TODAS AS MOEDAS
    # ========================================================

    def limpar_filtro(self):

        if self.imagem_original is None:
            messagebox.showwarning(
                "Atenção",
                "Selecione uma imagem primeiro."
            )
            return

        if self.resultado_yolo is None:
            messagebox.showwarning(
                "Atenção",
                "Faça uma detecção antes de limpar o filtro."
            )
            return

        self.filtro_atual = None

        # Restaura a imagem com todas as detecções do YOLO
        self.exibir_imagem(
            self.resultado_yolo,
            self.area_resultado
        )

        self.status_filtro.configure(
            text="Todos os filtros foram removidos."
        )

        self.resultado_filtro.configure(
            text="Filtro removido. Veja todas as moedas na aba Principal.",
            image=None
        )

        self.abas.set("Principal")

        self.status.configure(
            text=f"Total de moedas detectadas: {len(self.moedas_associadas)}"
        )

    # ========================================================
    # ALTERNAR TEMA
    # ========================================================

    def alternar_tema(self):

        if self.interruptor_tema.get():
            ctk.set_appearance_mode("dark")
            self.interruptor_tema.configure(
                text="Modo escuro"
            )
        else:
            ctk.set_appearance_mode("light")
            self.interruptor_tema.configure(
                text="Modo claro"
            )


# ============================================================
# EXECUTAR PROGRAMA
# ============================================================

if __name__ == "__main__":

    app = Aplicativo()
    app.mainloop()
