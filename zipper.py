import zipfile
import os

def zipdir(path, ziph, root_path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'backup_data.zip':
                continue
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, root_path)
            ziph.write(file_path, arcname)

if __name__ == '__main__':
    with zipfile.ZipFile('/home/dimas/gasss-predict/backup_data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('/home/dimas/gasss-predict', zipf, '/home/dimas')
        zipdir('/home/dimas/olah-data', zipf, '/home/dimas')
    print("Zip completed.")
