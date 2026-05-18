import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
import os
import torch


try:
    from main import load_resources, analyze_reviews
    from modules.model import predict_review
    from modules.antispam import compute_trust_score
except ImportError:
    print("Ошибка: Убедись, что gui.py лежит в одной папке с main.py")


class DiplomaApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Analyzer")
        self.geometry("980x850")
        ctk.set_appearance_mode("dark")

        # Ресурсы
        self.model, self.tokenizer, self.karta, self.device = (None, None, None, None)

        # Главный контейнер со скроллом
        self.main_frame = ctk.CTkScrollableFrame(self, width=940, height=820)
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Заголовок
        ctk.CTkLabel(self.main_frame, text="Анализ отзывов и проверка на спам", font=("Arial", 24, "bold")).pack(
            pady=15)

        self.load_btn = ctk.CTkButton(self.main_frame, text="Загрузить модель и данные",
                                      height=45, command=self.start_loading)
        self.load_btn.pack(pady=10)

        self.entry = ctk.CTkEntry(self.main_frame, placeholder_text="Введите текст отзыва...", width=700, height=45)
        self.entry.pack(pady=15)


        ctk.CTkLabel(self.main_frame, text="Результат одиночной проверки", font=("Arial", 14, "bold"),
                     text_color="#3a86ff").pack(pady=(5, 5))

        self.cards_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.cards_frame.pack(pady=5, padx=20)

        self.sent_card = ctk.CTkFrame(self.cards_frame, width=280, height=140, fg_color="#2b2b2b", corner_radius=10,
                                      border_width=1, border_color="#404040")
        self.sent_card.pack(side="left", padx=15)
        self.sent_card.pack_propagate(False)

        ctk.CTkLabel(self.sent_card, text="ТОНАЛЬНОСТЬ ОТЗЫВА", font=("Arial", 11, "bold"), text_color="gray").pack(
            pady=(12, 0))
        self.sent_status_label = ctk.CTkLabel(self.sent_card, text="ОЖИДАНИЕ", font=("Arial", 26, "bold"),
                                              text_color="gray")
        self.sent_status_label.pack(pady=(5, 0))
        self.sent_conf_label = ctk.CTkLabel(self.sent_card, text="Уверенность: 0.0%", font=("Arial", 11, "italic"),
                                            text_color="gray")
        self.sent_conf_label.pack(pady=(0, 10))

        self.trust_card = ctk.CTkFrame(self.cards_frame, width=280, height=140, fg_color="#2b2b2b", corner_radius=10,
                                       border_width=1, border_color="#404040")
        self.trust_card.pack(side="right", padx=15)
        self.trust_card.pack_propagate(False)

        ctk.CTkLabel(self.trust_card, text="ДОВЕРИЕ К ОТЗЫВУ", font=("Arial", 11, "bold"), text_color="gray").pack(
            pady=(12, 0))
        self.trust_score_label = ctk.CTkLabel(self.trust_card, text="0.0%", font=("Arial", 26, "bold"),
                                              text_color="gray")
        self.trust_score_label.pack(pady=(5, 0))
        self.trust_verdict_label = ctk.CTkLabel(self.trust_card, text="Ожидание анализа", font=("Arial", 11, "italic"),
                                                text_color="gray")
        self.trust_verdict_label.pack(pady=(0, 10))

        # ==========================================

        self.single_btn = ctk.CTkButton(self.main_frame, text="Анализировать текст",
                                        state="disabled", command=self.analyze_single)
        self.single_btn.pack(pady=15)

        ctk.CTkLabel(self.main_frame, text="—" * 65, text_color="gray").pack(pady=5)


        self.csv_btn = ctk.CTkButton(self.main_frame, text="Пакетная обработка CSV",
                                     state="disabled", fg_color="#2c6e49", command=self.analyze_csv)
        self.csv_btn.pack(pady=10)

        # ==========================================

        self.stats_frame = ctk.CTkFrame(self.main_frame, fg_color="#1e1e1e", corner_radius=12)
        self.stats_frame.pack(pady=15, padx=20, fill="x")

        ctk.CTkLabel(self.stats_frame, text="Статистика пакетной обработки выборки", font=("Arial", 16, "bold"),
                     text_color="#3a86ff").pack(pady=10)


        self.total_processed_label = ctk.CTkLabel(self.stats_frame, text="Обработано отзывов: 0 шт.",
                                                  font=("Arial", 13, "bold"))
        self.total_processed_label.pack(pady=(0, 5))

        self.csv_cards_layout = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        self.csv_cards_layout.pack(pady=10, padx=15)


        self.csv_sent_card = ctk.CTkFrame(self.csv_cards_layout, width=320, height=150, fg_color="#2b2b2b",
                                          corner_radius=10, border_width=1, border_color="#404040")
        self.csv_sent_card.pack(side="left", padx=15)
        self.csv_sent_card.pack_propagate(False)

        ctk.CTkLabel(self.csv_sent_card, text="ПРЕОБЛАДАЮЩАЯ ТОНАЛЬНОСТЬ", font=("Arial", 11, "bold"),
                     text_color="gray").pack(pady=(12, 0))
        self.csv_sent_status_label = ctk.CTkLabel(self.csv_sent_card, text="НЕТ ДАННЫХ", font=("Arial", 22, "bold"),
                                                  text_color="gray")
        self.csv_sent_status_label.pack(pady=(5, 0))
        self.csv_sent_details_label = ctk.CTkLabel(self.csv_sent_card,
                                                   text="Позитив: 0% | Негатив: 0%\nСр. уверенность: 0.0%",
                                                   font=("Arial", 11, "italic"), text_color="gray", justify="center")
        self.csv_sent_details_label.pack(pady=(2, 10))


        self.csv_trust_card = ctk.CTkFrame(self.csv_cards_layout, width=320, height=150, fg_color="#2b2b2b",
                                           corner_radius=10, border_width=1, border_color="#404040")
        self.csv_trust_card.pack(side="right", padx=15)
        self.csv_trust_card.pack_propagate(False)

        ctk.CTkLabel(self.csv_trust_card, text="ОБЩЕЕ ЗДОРОВЬЕ ВЫБОРКИ", font=("Arial", 11, "bold"),
                     text_color="gray").pack(pady=(12, 0))
        self.kpi_score_label = ctk.CTkLabel(self.csv_trust_card, text="0.0%", font=("Arial", 28, "bold"),
                                            text_color="gray")
        self.kpi_score_label.pack(pady=(5, 0))
        self.kpi_verdict_label = ctk.CTkLabel(self.csv_trust_card, text="Ожидание загрузки CSV",
                                              font=("Arial", 11, "italic"), text_color="gray")
        self.kpi_verdict_label.pack(pady=(0, 10))

        # ==========================================


        self.result_text = ctk.CTkTextbox(self.main_frame, width=800, height=150, font=("Consolas", 14))
        self.result_text.pack(pady=15)
        self.result_text.insert("0.0", "Система готова к работе.")

    def start_loading(self):
        if not os.path.exists('weights/best_model.pth') or not os.path.exists('data/karta_lookup.pkl'):
            messagebox.showerror("Файлы не найдены",
                                 "Отсутствуют необходимые файлы:\n- weights/best_model.pth\n- data/karta_lookup.pkl\n\nСначала запустите train.py!")
            return

        self.load_btn.configure(text="Загрузка...", state="disabled")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "Начинаю загрузку ресурсов...\n")
        threading.Thread(target=self.init_resources, daemon=True).start()

    def init_resources(self):
        try:
            resources = load_resources()
            self.model, self.tokenizer, self.karta, self.device = resources
            self.after(0, self.success_ui)
        except Exception as err:
            msg = str(err)
            self.after(0, lambda m=msg: self.error_ui(m))

    def success_ui(self):
        self.load_btn.configure(text="Модель готова ✓", fg_color="green")
        self.single_btn.configure(state="normal")
        self.csv_btn.configure(state="normal")
        self.result_text.insert("end", f"Успешно загружено! Устройство: {self.device}\n")

    def error_ui(self, message):
        self.load_btn.configure(text="Ошибка", state="normal", fg_color="red")
        messagebox.showerror("Ошибка загрузки", f"Проблема при чтении файлов:\n{message}")
        self.result_text.insert("end", f"ОШИБКА: {message}\n")

    def analyze_single(self):
        text = self.entry.get().strip()
        if not text: return

        try:
            sentiment, conf, aspects = predict_review(text, self.model, self.tokenizer, self.device)

            trust_score, _ = compute_trust_score(text, self.karta, sim_score=1.0)

            if trust_score <= 1.0:
                trust_display = trust_score * 100
            else:
                trust_display = trust_score

            res = f"--- РЕЗУЛЬТАТ ---\n"
            res += f"Тональность: {'ПОЗИТИВ' if sentiment == 'positive' else 'НЕГАТИВ'} ({conf * 100:.1f}%)\n"
            res += f"Доверие: {trust_display:.1f}%\n"

            asp_names = [a['aspect'] for a in aspects] if aspects else ["—"]
            res += f"Аспекты: {', '.join(asp_names)}\n"
            res += f"-----------------\n\n"


            if sentiment == 'positive':
                self.sent_status_label.configure(text="ПОЗИТИВ", text_color="#2c6e49")
                self.sent_card.configure(border_color="#2c6e49")
            else:
                self.sent_status_label.configure(text="НЕГАТИВ", text_color="#b7094c")
                self.sent_card.configure(border_color="#b7094c")
            self.sent_conf_label.configure(text=f"Уверенность: {conf * 100:.1f}%")


            self.trust_score_label.configure(text=f"{trust_display:.1f}%")
            if trust_display > 50:
                self.trust_score_label.configure(text_color="#2c6e49")
                self.trust_verdict_label.configure(text="БЕЗОПАСНЫЙ ОТЗЫВ", text_color="#2c6e49")
                self.trust_card.configure(border_color="#2c6e49")
            else:
                self.trust_score_label.configure(text_color="#b7094c")
                self.trust_verdict_label.configure(text="ПОДОЗРИТЕЛЬНЫЙ (СПАМ)", text_color="#b7094c")
                self.trust_card.configure(border_color="#b7094c")

            self.result_text.insert("1.0", res)
        except Exception as e:
            messagebox.showerror("Ошибка анализа", str(e))

    def analyze_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path: return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path: return

        self.csv_btn.configure(state="disabled", text="Обработка файла...")

        def process():
            try:
                df_in = pd.read_csv(file_path)
                if 'text' not in df_in.columns:
                    self.after(0, lambda: messagebox.showerror("Ошибка", "В CSV нет колонки 'text'"))
                    return

                texts = df_in['text'].dropna().tolist()
                res_df = analyze_reviews(texts, self.model, self.tokenizer, self.karta, self.device)
                res_df.to_csv(save_path, index=False, encoding='utf-8-sig')

                total = len(res_df)

                sent_col = 'sentiment' if 'sentiment' in res_df.columns else res_df.columns[1]
                conf_col = 'confidence' if 'confidence' in res_df.columns else res_df.columns[2]
                trust_col = 'trust_score' if 'trust_score' in res_df.columns else res_df.columns[3]

                pos_count = len(res_df[res_df[sent_col] == 'positive'])
                neg_count = total - pos_count
                spam_count = len(res_df[res_df[trust_col] < 50])

                avg_conf = res_df[conf_col].mean()
                if res_df[trust_col].max() <= 1.0:
                    res_df[trust_col] = res_df[trust_col] * 100

                pos_share = pos_count / total if total > 0 else 0
                neg_share = neg_count / total if total > 0 else 0
                spam_share = spam_count / total if total > 0 else 0
                health_share = 1.0 - spam_share


                self.after(0, lambda: self.update_csv_stats_ui(total, pos_share, neg_share, avg_conf, health_share))
                self.after(0,
                           lambda: messagebox.showinfo("Готово", "Файл сохранен!\nМетрики ML и Антиспама обновлены."))

            except Exception as e:
                self.after(0, lambda err=str(e): messagebox.showerror("Ошибка CSV", err))
            finally:
                self.after(0, lambda: self.csv_btn.configure(state="normal", text="Пакетная обработка CSV"))

        threading.Thread(target=process, daemon=True).start()

    def update_csv_stats_ui(self, total, pos_share, neg_share, avg_conf, health_share):
        """ Метод отрисовки макро-карточек для результатов пакетного анализа CSV """
        self.total_processed_label.configure(text=f"Обработано отзывов: {total} шт.")


        conf_multiplier = 100 if avg_conf <= 1.0 else 1
        avg_conf_pct = avg_conf * conf_multiplier

        if pos_share >= neg_share:
            self.csv_sent_status_label.configure(text="ПОЗИТИВНАЯ", text_color="#2c6e49")
            self.csv_sent_card.configure(border_color="#2c6e49")
        else:
            self.csv_sent_status_label.configure(text="НЕГАТИВНАЯ", text_color="#b7094c")
            self.csv_sent_card.configure(border_color="#b7094c")

        details_text = f"Позитив: {pos_share * 100:.1f}% | Негатив: {neg_share * 100:.1f}%\nСр. уверенность BERT: {avg_conf_pct:.1f}%"
        self.csv_sent_details_label.configure(text=details_text, text_color="white")

        
        health_percentage = health_share * 100
        self.kpi_score_label.configure(text=f"{health_percentage:.1f}%")

        if health_percentage >= 80:
            self.kpi_score_label.configure(text_color="#2c6e49")
            self.kpi_verdict_label.configure(text="СПАМ НЕ ОБНАРУЖЕН", text_color="#2c6e49")
            self.csv_trust_card.configure(border_color="#2c6e49")
        elif 50 <= health_percentage < 80:
            self.kpi_score_label.configure(text_color="#d48c00")
            self.kpi_verdict_label.configure(text="УМЕРЕННЫЕ АНОМАЛИИ", text_color="#d48c00")
            self.csv_trust_card.configure(border_color="#d48c00")
        else:
            self.kpi_score_label.configure(text_color="#b7094c")
            self.kpi_verdict_label.configure(text="ВЫСОКИЙ УРОВЕНЬ СПАМА", text_color="#b7094c")
            self.csv_trust_card.configure(border_color="#b7094c")


if __name__ == "__main__":
    app = DiplomaApp()
    app.mainloop()