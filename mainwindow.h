#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStandardItemModel>  // Tambahkan ini

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnImport_clicked();
    void on_btnTrain_clicked();
    void on_runButton_clicked();
    void importCSV();

private:
    Ui::MainWindow *ui;
    QStandardItemModel *model;  // Tambahkan ini
    QString currentFile;
    QString csvFilePath;
};

#endif // MAINWINDOW_H