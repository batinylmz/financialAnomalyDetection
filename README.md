# financialAnomalyDetection
Project where we benchmarked models using financial data.


# 📈 Financial Anomaly Detection with AI

Bu proje, finansal veriler üzerindeki şüpheli hareketleri ve anomali (aykırı değer) durumlarını tespit etmek için Makine Öğrenmesi tekniklerini kullanır. Özellikle piyasa verilerindeki sert ve olağandışı değişimleri yakalamayı hedefler.

## 🚀 Kullanılan Teknolojiler ve Kütüphaneler

Proje Python dili ile geliştirilmiştir ve aşağıdaki kütüphaneleri kullanır:

* **PyOD:** Gelişmiş anomali tespiti modelleri için (Isolation Forest vb.).
* **Scikit-Learn:** Model değerlendirme ve metrikler (F1-Score, ROC AUC).
* **Pandas & Numpy:** Veri manipülasyonu ve analizi.
* **Itertools:** Hiperparametre optimizasyonu.

## 🛠️ Kurulum

Projeyi yerel bilgisayarınızda (macOS/Windows/Linux) çalıştırmak için önce gerekli paketleri yüklemelisiniz:

```bash
pip install numpy pandas scikit-learn pyod