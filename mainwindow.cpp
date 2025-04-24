#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "rustcore_bridge.h"
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QTableWidgetItem>
#include <QProcess>
#include <QMessageBox>
    
extern "C" {
    char* train_model(const char* filename);
    void free_string(char* s);
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
        ui->setupUi(this);
        model = new QStandardItemModel(this);

    connect(ui->btnImport, &QPushButton::clicked, this, &MainWindow::on_btnImport_clicked);
    connect(ui->runButton, &QPushButton::clicked, this, &MainWindow::on_runButton_clicked);
    connect(ui->btnImport, &QPushButton::clicked, this, &MainWindow::importCSV);
    ui->tableWidget->setRowCount(0);
    ui->tableWidget->setColumnCount(0);
}

MainWindow::~MainWindow() {
    delete ui;
}
void MainWindow::on_btnImport_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, 
        "Open CSV File", "", "CSV Files (*.csv)");
    
    if(filename.isEmpty()) return;
    
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Error", "Could not open file");
        return;
    }
    
    model->clear();
    
    QTextStream in(&file);
    bool firstLine = true;
    
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList fields = line.split(",");
        
        if(firstLine) {
            model->setHorizontalHeaderLabels(fields);
            firstLine = false;
            continue;
        }
        QList<QStandardItem*> rowItems;
        for(const QString &field : fields) {
            rowItems.append(new QStandardItem(field));
        }
        model->appendRow(rowItems);
    }
    
    currentFile = filename;
    ui->lblStatus->setText(QString("Loaded: %1").arg(filename));
}

void MainWindow::on_btnTrain_clicked()
{
    if(currentFile.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please import a CSV file first");
        return;
    }
    
    QByteArray ba = currentFile.toUtf8();
    const char *c_filename = ba.constData();
    
    char* result = train_model(c_filename);
    QString qResult = QString::fromUtf8(result);
    
    ui->txtResult->setPlainText(qResult);
    free_string(result);
}

void MainWindow::on_runButton_clicked() {
    ui->textEdit->clear();
    
    QByteArray ba = csvFilePath.toUtf8();
    const char *c_filename = ba.constData();

    char* result = train_model(c_filename);
    QString imagePath = "output.png";

    // Tampilkan gambar grafik hasil training
    QPixmap pix(imagePath);
    if (!pix.isNull()) {
        ui->imageLabel->setPixmap(pix.scaled(640, 480, Qt::KeepAspectRatio));
        ui->textEdit->append("[Gambar berhasil dimuat]");
    } else {
        ui->textEdit->append("[Gambar tidak ditemukan]");
    }
    QProcess *process = new QProcess(this);

#if defined(Q_OS_WIN)
    QString program = "target\\release\\neural_net_rust.exe";  // Sesuaikan dengan nama binary
#else
    QString program = "./target/release/neural_net_rust";
#endif

    QStringList arguments;
    if (!csvFilePath.isEmpty()) {
        arguments << csvFilePath; // Tambahkan path CSV jika sudah ada
    } else {
        QMessageBox::warning(this, "No File", "Silakan impor file CSV terlebih dahulu.");
        return;
    }

    process->start(program, arguments);

    connect(process, &QProcess::readyReadStandardOutput, [=]() {
        QByteArray output = process->readAllStandardOutput();
        ui->textEdit->append(QString::fromLocal8Bit(output));
    });

    connect(process, &QProcess::readyReadStandardError, [=]() {
        QByteArray err = process->readAllStandardError();
        ui->textEdit->append("[ERROR] " + QString::fromLocal8Bit(err));
    });

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            [=](int exitCode, QProcess::ExitStatus){
        ui->textEdit->append(QString("Proses selesai dengan kode %1").arg(exitCode));
    });

}

void MainWindow::importCSV() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open CSV File", "", "CSV files (*.csv);;All files (*.*)");
    if (filePath.isEmpty())
        return;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Error", "Tidak bisa membuka file.");
        return;
    }

    csvFilePath = filePath;

    QTextStream in(&file);
    ui->tableWidget->clear();
    ui->tableWidget->setRowCount(0);

    // Baca baris pertama sebagai header
    QString headerLine = in.readLine();
    QStringList headers = headerLine.split(',');

    ui->tableWidget->setColumnCount(headers.size());
    ui->tableWidget->setHorizontalHeaderLabels(headers);

    int row = 0;
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList fields = line.split(',');

        ui->tableWidget->insertRow(row);
        for (int col = 0; col < fields.size(); ++col) {
            QTableWidgetItem *item = new QTableWidgetItem(fields[col]);
            ui->tableWidget->setItem(row, col, item);
        }
        row++;
    }

    file.close();
}


