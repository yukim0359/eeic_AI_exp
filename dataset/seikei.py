import os

# ディレクトリパス
easy_directory = 'data/dataset_blur_5'
medium_directory = 'data/dataset_blur_yoko'
hard_directory = 'data/dataset_blur_15'

label_directory = 'data/label'

file_names = []

# ディレクトリ内のファイルを取得
easy_files = os.listdir(easy_directory)

for index, file_name in enumerate(easy_files):
    # 拡張子が .png のファイルのみを処理
    if file_name.endswith('.png'):
        # 拡張子を取り除いたファイル名を取得
        name_without_extension = file_name[:-4]  # '.png'を取り除く

        # 新しいファイル名を生成
        new_name = f"{index + 1}.png"

        # 元のファイルのフルパス
        original_file_path = os.path.join(easy_directory, file_name)

        # 新しいファイルのフルパス
        new_file_path = os.path.join(easy_directory, new_name)

        # 拡張子を取り除いたファイル名をリストに追加
        file_names.append(name_without_extension)

        # ファイル名を変更
        os.rename(original_file_path, new_file_path)

        # ラベルファイルを保存
        label_file_path = os.path.join(label_directory, f"{index + 1}.txt")
        with open(label_file_path, 'w') as label_file:
            label_file.write(name_without_extension)

for i in range(len(file_names)):
    file_name_without_png = file_names[i]

    new_name = f"{i + 1}.png"

    original_file_path = os.path.join(medium_directory, file_name_without_png + '.png')
    new_file_path = os.path.join(medium_directory, new_name)
    os.rename(original_file_path, new_file_path)

    original_file_path = os.path.join(hard_directory, file_name_without_png + '.png')
    new_file_path = os.path.join(hard_directory, new_name)
    os.rename(original_file_path, new_file_path)