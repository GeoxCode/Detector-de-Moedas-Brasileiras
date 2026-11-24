
import os
import re
import cv2
import numpy as np
import customtkinter as ctk

from tkinter import filedialog, messagebox
from PIL import Image
from ultralytics import YOLO


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODELO_PATH = r"C:\Users\Gisela Caroline\Desktop\Code_Legal\runs\detector_valores\weights\best.pt"

VALORES_MOEDAS = (5, 10, 25, 50, 100)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================================================
# APLICAÇÃO
# ============================================================

class Aplicativo(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Detector de Moedas Brasileiras - YOLO26")
        self.geometry("1200x800")
        self.minsize(950, 650)

        self.caminho_imagem = None
        self.imagem_original = None
        self.imagem_processada = None

        self.resultado_yolo = None
        self.moedas_associadas = []
        self.filtro_atual = None

        # Referências das imagens exibidas
        self.imagem_tk = None
        self.imagem_original_tk = None
        self.imagem_processada_tk = None

        # Carregamento do modelo
        if not os.path.isfile(MODELO_PATH):
            messagebox.showerror(
                "Erro",
                f"Modelo não encontrado:\n{MODELO_PATH}"
            )
            self.destroy()
            return

        try:
            self.modelo = YOLO(MODELO_PATH)

        except Exception as erro:
            messagebox.showerror(
                "Erro ao carregar modelo",
                str(erro)
            )
            self.destroy()
            return

        self.construir_interface()

    # ========================================================
    # INTERFACE
    # ========================================================

    def construir_interface(self):

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
            width=230
        )
        self.painel_lateral.grid(
            row=0,
            column=0,
            sticky="ns",
            padx=(0, 12)
        )
        self.painel_lateral.grid_propagate(False)

        titulo = ctk.CTkLabel(
            self.painel_lateral,
            text="DETECTOR\nDE MOEDAS",
            font=("Arial", 24, "bold"),
            text_color="#D4A017"
        )
        titulo.pack(pady=(25, 20))

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

        self.titulo_previa = ctk.CTkLabel(
            self.painel_lateral,
            text="Imagem selecionada",
            font=("Arial", 14, "bold")
        )
        self.titulo_previa.pack(pady=(20, 5))

        self.previa = ctk.CTkLabel(
            self.painel_lateral,
            text="Nenhuma imagem\nselecionada",
            width=200,
            height=150
        )
        self.previa.pack(padx=10, pady=5)

        self.interruptor_tema = ctk.CTkSwitch(
            self.painel_lateral,
            text="Modo escuro",
            command=self.alternar_tema
        )
        self.interruptor_tema.select()
        self.interruptor_tema.pack(pady=25)

        # Abas
        self.abas = ctk.CTkTabview(self.conteudo)
        self.abas.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.aba_principal = self.abas.add("Principal")
        self.abafiltro = self.abas.add("Filtro")
        self.aba_processamento = self.abas.add(
            "Pré-processamento"
        )

        self.construir_aba_principal()
        self.construir_aba_filtro()
        self.construir_aba_processamento()

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

        self.aba_principal.grid_columnconfigure(0, weight=1)
        self.aba_principal.grid_rowconfigure(1, weight=1)

        self.titulo_resultado = ctk.CTkLabel(
            self.aba_principal,
            text="Resultado da Detecção YOLO",
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
    # ABA DE PRÉ-PROCESSAMENTO
    # ========================================================

    def construir_aba_processamento(self):

        self.aba_processamento.grid_columnconfigure(
            (0, 1),
            weight=1
        )
        self.aba_processamento.grid_rowconfigure(
            2,
            weight=1
        )

        titulo = ctk.CTkLabel(
            self.aba_processamento,
            text="Pré-processamento de Imagens",
            font=("Arial", 21, "bold")
        )
        titulo.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=15
        )

        # Seleção do filtro
        self.selecao_filtro = ctk.CTkOptionMenu(
            self.aba_processamento,
            values=[
                "Segmentação",
                "Tons de cinza",
                "CLAHE"
            ],
            command=self.preparar_filtro
        )
        self.selecao_filtro.grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            sticky="ew"
        )

        self.botao_aplicar = ctk.CTkButton(
            self.aba_processamento,
            text="Aplicar pré-processamento",
            command=self.aplicar_preprocessamento
        )
        self.botao_aplicar.grid(
            row=1,
            column=1,
            padx=10,
            pady=10,
            sticky="ew"
        )

        # Área de comparação
        quadro_original = ctk.CTkFrame(
            self.aba_processamento
        )
        quadro_original.grid(
            row=2,
            column=0,
            padx=8,
            pady=10,
            sticky="nsew"
        )

        quadro_processado = ctk.CTkFrame(
            self.aba_processamento
        )
        quadro_processado.grid(
            row=2,
            column=1,
            padx=8,
            pady=10,
            sticky="nsew"
        )

        ctk.CTkLabel(
            quadro_original,
            text="Imagem original",
            font=("Arial", 16, "bold")
        ).pack(pady=10)

        ctk.CTkLabel(
            quadro_processado,
            text="Imagem processada",
            font=("Arial", 16, "bold")
        ).pack(pady=10)

        self.preview_original = ctk.CTkLabel(
            quadro_original,
            text="Selecione uma imagem",
            height=350
        )
        self.preview_original.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        self.preview_processada = ctk.CTkLabel(
            quadro_processado,
            text="Aguardando processamento",
            height=350
        )
        self.preview_processada.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5
        )

        # Informação sobre o método
        self.info_processamento = ctk.CTkLabel(
            self.aba_processamento,
            text="Escolha um método para visualizar o processamento.",
            font=("Arial", 13),
            wraplength=700
        )
        self.info_processamento.grid(
            row=3,
            column=0,
            columnspan=2,
            pady=10
        )

        self.botao_original = ctk.CTkButton(
            self.aba_processamento,
            text="Restaurar imagem original",
            command=self.restaurar_original
        )
        self.botao_original.grid(
            row=4,
            column=0,
            padx=10,
            pady=10,
            sticky="ew"
        )

        self.botao_usar = ctk.CTkButton(
            self.aba_processamento,
            text="Usar imagem processada",
            command=self.confirmar_processamento,
            fg_color="#228B22",
            hover_color="#176B17"
        )
        self.botao_usar.grid(
            row=4,
            column=1,
            padx=10,
            pady=10,
            sticky="ew"
        )

    # ========================================================
    # PREPARAR FILTRO
    # ========================================================

    def preparar_filtro(self, escolha):

        descricoes = {
            "Segmentação":
                "Agrupa pixels semelhantes em regiões usando K-Means.",
            "Tons de cinza":
                "Remove as cores e mantém as informações de intensidade.",
            "CLAHE":
                "Melhora o contraste local e destaca detalhes."
        }

        self.info_processamento.configure(
            text=descricoes.get(escolha, "")
        )

    # ========================================================
    # APLICAR PRÉ-PROCESSAMENTO
    # ========================================================

    def aplicar_preprocessamento(self):

        if self.imagem_original is None:
            messagebox.showwarning(
                "Atenção",
                "Selecione uma imagem primeiro."
            )
            return

        escolha = self.selecao_filtro.get()

        imagem = self.imagem_original.copy()

        try:

            # -----------------------------------------------
            # TONS DE CINZA
            # -----------------------------------------------

            if escolha == "Tons de cinza":

                cinza = cv2.cvtColor(
                    imagem,
                    cv2.COLOR_BGR2GRAY
                )

                # Retorna para três canais para o YOLO
                processada = cv2.cvtColor(
                    cinza,
                    cv2.COLOR_GRAY2BGR
                )

            # -----------------------------------------------
            # CLAHE
            # -----------------------------------------------

            elif escolha == "CLAHE":

                lab = cv2.cvtColor(
                    imagem,
                    cv2.COLOR_BGR2LAB
                )

                canal_l, canal_a, canal_b = cv2.split(lab)

                clahe = cv2.createCLAHE(
                    clipLimit=2.0,
                    tileGridSize=(8, 8)
                )

                canal_l = clahe.apply(canal_l)

                imagem_clahe = cv2.merge(
                    (canal_l, canal_a, canal_b)
                )

                processada = cv2.cvtColor(
                    imagem_clahe,
                    cv2.COLOR_LAB2BGR
                )

            # -----------------------------------------------
            # THRESHOLD
            # -----------------------------------------------

            elif escolha == "Threshold":

                cinza = cv2.cvtColor(
                    imagem,
                    cv2.COLOR_BGR2GRAY
                )

                _, resultado = cv2.threshold(
                    cinza,
                    127,
                    255,
                    cv2.THRESH_BINARY
                )

                processada = cv2.cvtColor(
                    resultado,
                    cv2.COLOR_GRAY2BGR
                )

            # -----------------------------------------------
            # OTSU
            # -----------------------------------------------

            elif escolha == "Otsu":

                cinza = cv2.cvtColor(
                    imagem,
                    cv2.COLOR_BGR2GRAY
                )

                _, resultado = cv2.threshold(
                    cinza,
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                processada = cv2.cvtColor(
                    resultado,
                    cv2.COLOR_GRAY2BGR
                )

            # -----------------------------------------------
            # SEGMENTAÇÃO (K-MEANS)
            # -----------------------------------------------

            elif escolha == "Segmentação":

                altura, largura = imagem.shape[:2]

                # Reduz temporariamente a imagem para
                # acelerar o processamento
                escala = min(
                    1.0,
                    600 / max(altura, largura)
                )

                if escala < 1.0:
                    reduzida = cv2.resize(
                        imagem,
                        None,
                        fx=escala,
                        fy=escala,
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    reduzida = imagem.copy()

                pixels = reduzida.reshape(
                    (-1, 3)
                ).astype(np.float32)

                criterios = (
                    cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER,
                    100,
                    0.2
                )

                k = 3

                _, rotulos, centros = cv2.kmeans(
                    pixels,
                    k,
                    None,
                    criterios,
                    5,
                    cv2.KMEANS_PP_CENTERS
                )

                centros = np.uint8(centros)

                segmentada = centros[
                    rotulos.flatten()
                ]

                segmentada = segmentada.reshape(
                    reduzida.shape
                )

                # Retorna ao tamanho original
                processada = cv2.resize(
                    segmentada,
                    (largura, altura),
                    interpolation=cv2.INTER_NEAREST
                )

            else:
                processada = imagem.copy()

            self.imagem_processada = processada

            # Atualiza as duas imagens lado a lado
            self.exibir_imagem(
                self.imagem_original,
                self.preview_original,
                tamanho=(420, 380)
            )

            self.exibir_imagem(
                self.imagem_processada,
                self.preview_processada,
                tamanho=(420, 380)
            )

            self.info_processamento.configure(
                text=f"Método aplicado: {escolha}"
            )

            self.status.configure(
                text=f"Pré-processamento aplicado: {escolha}"
            )

        except Exception as erro:

            messagebox.showerror(
                "Erro no processamento",
                str(erro)
            )

    # ========================================================
    # CONFIRMAR IMAGEM PROCESSADA
    # ========================================================

    def confirmar_processamento(self):

        if self.imagem_processada is None:
            messagebox.showwarning(
                "Atenção",
                "Aplique um filtro primeiro."
            )
            return

        self.status.configure(
            text="Imagem processada selecionada para o YOLO."
        )

        messagebox.showinfo(
            "Imagem selecionada",
            "A imagem processada será utilizada na próxima detecção YOLO."
        )

    # ========================================================
    # RESTAURAR IMAGEM ORIGINAL
    # ========================================================

    def restaurar_original(self):

        if self.imagem_original is None:
            messagebox.showwarning(
                "Atenção",
                "Selecione uma imagem primeiro."
            )
            return

        self.imagem_processada = self.imagem_original.copy()

        self.exibir_imagem(
            self.imagem_original,
            self.preview_original,
            tamanho=(420, 380)
        )

        self.exibir_imagem(
            self.imagem_processada,
            self.preview_processada,
            tamanho=(420, 380)
        )

        self.info_processamento.configure(
            text="Imagem original restaurada. Nenhum pré-processamento está aplicado."
        )

        self.status.configure(
            text="Imagem original selecionada para detecção."
        )

    # ========================================================
    # ABA DE FILTROS DE MOEDAS
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

        for indice, centavos in enumerate(VALORES_MOEDAS):

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
    # NORMALIZAR NOMES
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
                "Não foi possível abrir a imagem."
            )
            return

        self.caminho_imagem = caminho
        self.imagem_original = imagem
        self.imagem_processada = imagem.copy()

        self.resultado_yolo = None
        self.moedas_associadas = []
        self.filtro_atual = None

        self.selecao_filtro.set("Tons de cinza")

        self.exibir_imagem(
            imagem,
            self.previa,
            tamanho=(200, 150)
        )

        self.exibir_imagem(
            imagem,
            self.preview_original,
            tamanho=(420, 380)
        )

        self.exibir_imagem(
            imagem,
            self.preview_processada,
            tamanho=(420, 380)
        )

        self.area_resultado.configure(
            image=None,
            text="Imagem carregada. Execute a detecção."
        )

        self.resultado_filtro.configure(
            image=None,
            text="Faça uma detecção para utilizar os filtros."
        )

        self.info_processamento.configure(
            text="Imagem carregada. Escolha um método."
        )

        self.status_filtro.configure(
            text="Selecione o valor da moeda"
        )

        self.status.configure(
            text="Imagem carregada com sucesso."
        )

    # ========================================================
    # EXIBIR IMAGEM
    # ========================================================

    def exibir_imagem(
        self,
        imagem,
        componente,
        tamanho=(650, 450)
    ):

        if imagem is None:
            return

        imagem_rgb = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )

        imagem_pil = Image.fromarray(imagem_rgb)

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

        if self.imagem_processada is None:
            self.imagem_processada = self.imagem_original.copy()

        try:

            self.status.configure(
                text="Executando detecção YOLO..."
            )
            self.update_idletasks()

            # O YOLO recebe a imagem processada
            imagem_entrada = self.imagem_processada.copy()

            resultados = self.modelo.predict(
                source=imagem_entrada,
                conf=0.25,
                verbose=False
            )

            resultado = resultados[0]

            self.resultado_yolo = resultado.plot()

            nomes = self.modelo.names
            deteccoes = []

            for caixa in resultado.boxes:

                classe_id = int(caixa.cls[0].item())

                nome_classe = nomes[classe_id]

                x1, y1, x2, y2 = (
                    int(valor)
                    for valor in caixa.xyxy[0].tolist()
                )

                confianca = float(
                    caixa.conf[0].item()
                )

                deteccoes.append({
                    "nome": str(nome_classe),
                    "nome_normalizado": self.normalizar_nome(
                        nome_classe
                    ),
                    "caixa": (x1, y1, x2, y2),
                    "confianca": confianca
                })

            # Caixas da classe Moeda
            caixas_moedas = [
                d for d in deteccoes
                if d["nome_normalizado"] == "moeda"
            ]

            # Classes de valores
            classes_valores = {
                self.normalizar_nome(
                    f"{valor} Centavos"
                ): valor
                for valor in VALORES_MOEDAS
            }

            deteccoes_valores = [
                d for d in deteccoes
                if d["nome_normalizado"] in classes_valores
            ]

            self.moedas_associadas = []

            # Associação das caixas de valores às moedas
            for moeda in caixas_moedas:

                x1, y1, x2, y2 = moeda["caixa"]

                melhor_valor = None
                melhor_confianca = -1

                for deteccao in deteccoes_valores:

                    vx1, vy1, vx2, vy2 = deteccao["caixa"]

                    centro_x = (vx1 + vx2) / 2
                    centro_y = (vy1 + vy2) / 2

                    dentro = (
                        x1 <= centro_x <= x2
                        and y1 <= centro_y <= y2
                    )

                    if dentro:
                        if deteccao["confianca"] > melhor_confianca:

                            melhor_confianca = deteccao["confianca"]
                            melhor_valor = deteccao["nome"]

                self.moedas_associadas.append({
                    "caixa": moeda["caixa"],
                    "valor": melhor_valor
                })

            # Mostra resultado
            self.exibir_imagem(
                self.resultado_yolo,
                self.area_resultado
            )

            self.filtro_atual = None

            total = len(self.moedas_associadas)

            contagem = {}

            for valor in VALORES_MOEDAS:

                contagem[valor] = sum(
                    1 for moeda in self.moedas_associadas
                    if moeda["valor"] is not None
                    and self.normalizar_nome(moeda["valor"])
                    == self.normalizar_nome(
                        f"{valor} Centavos"
                    )
                )

            texto = " | ".join(
                f"{valor}¢: {contagem[valor]}"
                for valor in VALORES_MOEDAS
            )

            metodo = self.selecao_filtro.get()

            self.status.configure(
                text=(
                    f"Total de moedas: {total}\n"
                    f"{texto}\n"
                    f"Imagem utilizada: {metodo}"
                )
            )

            self.status_filtro.configure(
                text="Escolha um valor para filtrar as moedas."
            )

            self.resultado_filtro.configure(
                image=None,
                text="Selecione um dos valores acima."
            )

        except Exception as erro:

            messagebox.showerror(
                "Erro na detecção",
                str(erro)
            )

            self.status.configure(
                text="Erro durante a detecção."
            )

    # ========================================================
    # FILTRAR POR VALOR
    # ========================================================

    def selecionar_filtro(self, valor):

        if self.resultado_yolo is None:
            messagebox.showwarning(
                "Atenção",
                "Execute a detecção primeiro."
            )
            return

        self.filtro_atual = valor

        # Usa a imagem que foi enviada ao YOLO
        imagem_filtrada = self.imagem_processada.copy()

        filtro_normalizado = self.normalizar_nome(
            f"{valor} Centavos"
        )

        moedas_filtradas = [
            moeda for moeda in self.moedas_associadas
            if moeda["valor"] is not None
            and self.normalizar_nome(moeda["valor"])
            == filtro_normalizado
        ]

        # Desenha somente uma caixa por moeda
        # Não desenha nomes nem etiquetas
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

        quantidade = len(moedas_filtradas)

        self.status_filtro.configure(
            text=f"Filtro: {valor} Centavos"
        )

        self.resultado_filtro.configure(
            text=""
        )

        self.abas.set("Filtro")

        self.status.configure(
            text=(
                f"Moedas de {valor} centavos encontradas: "
                f"{quantidade}"
            )
        )

    # ========================================================
    # MOSTRAR TODAS
    # ========================================================

    def limpar_filtro(self):

        if self.resultado_yolo is None:
            messagebox.showwarning(
                "Atenção",
                "Execute a detecção primeiro."
            )
            return

        self.filtro_atual = None

        self.exibir_imagem(
            self.resultado_yolo,
            self.area_resultado
        )

        self.resultado_filtro.configure(
            image=None,
            text="Todos os filtros foram removidos."
        )

        self.status_filtro.configure(
            text="Todos os filtros foram removidos."
        )

        self.abas.set("Principal")

        self.status.configure(
            text=(
                f"Total de moedas detectadas: "
                f"{len(self.moedas_associadas)}"
            )
        )

    # ========================================================
    # MODO CLARO E ESCURO
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
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    app = Aplicativo()
    app.mainloop()
