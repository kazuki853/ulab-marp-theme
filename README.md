# UchidaLab Marp Themes
内田研のゼミ・輪講・プロジェクト研究での発表資料作成に利用できる[Marp](https://marp.app/)テーマです。
HTMLを書くことで利用できる独自の機能をいくつか追加しています。

## 導入方法
1. このリポジトリをローカルにクローンします。
    ```
    git clone https://github.com/kazuki853/ulab-marp-theme.git
    ```
2. MarkdownファイルをHTML / CSS, PDF, PowerPoint , 画像形式に簡単に変換することができる[Marp CLI](https://github.com/marp-team/marp-cli)をインストールします。（詳細なインストール方法はリンク先で確認してください。）
 
3. VS Codeの拡張機能として[Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode)をインストールします。
    作成したスライドの内容をvscode上で確認しながら作業をすすめることができます。

4. CSSのパスをVS Codeの拡張機能に認識させます。VS Code上で![VS Codeの設定アイコン](https://cz-cdn.shoeisha.jp/static/images/article/16930/16930_015a.png)をクリックまたは、[Command］＋［,］または、［Ctrl］＋［,］で設定を開くことができます。

    Markdown > Marp: Themesを開きましょう。Marp for VS Codeがしっかりとインストールできていれば開くことができるはずです。<img width="1065" alt="Marp for VS Code: Marp: Themes" src="https://user-images.githubusercontent.com/59504885/236084945-39b3bdbb-00b8-4e40-96e0-de215d5e378c.png">

    項目の追加から、CSSファイルがおいてある場所へのパスを書きましょう。下のように相対パスで書いても構いませんし、絶対パスで書いても構いません。

    ulab-marp-themeのディレクトリにいる場合、相対パスで記述すると以下のようになります。
    
    ```
    ./themes/theme_E.css
    ./themes/theme_U.css
    ```
    <img width="1058" alt="Marp for VS Code: Marp: Themes 項目の追加" src="https://user-images.githubusercontent.com/59504885/236085433-9a8e978b-e12b-4260-a090-1f104ce97a85.png">
    
    ⚠注意⚠ CSSファイルを置く場所を変更するとうまくテーマが読み込めなくなります。その場合にはパスを再度設定し直してください。

5. `samples/sample.md`をVS Codeで開き右上のプレビューを押すと以下のように表示されるはずです。

    ![VS-Code-Marp-Ulab-Theme](https://user-images.githubusercontent.com/59504885/236092381-e66af29a-aab9-4b64-8216-5d85b066107e.gif)
    
6. PDF形式など他の形式にエクスポートしたい場合には次のように`Export Slide Deck`をクリックします。

    ![VS-Code-Marp-Ulab-Theme-export-toggle](https://user-images.githubusercontent.com/59504885/236094804-aecb073c-a442-4e9b-973e-c51a0198f64a.gif)
    
7. 使用するテーマはMarkdownのはじめに書かれている内容を書き換えることで変更することができます。

    ```
    ---
    marp: true
    theme: theme_E
    paginate: true
    header: ここにヘッダーの内容を書きます
    footer: ここにフッターの内容を書きます
    math: katex
    ---
    ```
    デフォルトだと上のようになっていますが、`theme`の値を`theme_U`にすることで別のテーマを適用することができます。


## 各テーマの説明
### theme_E

輪講などで使用しているテーマです。緑を基調としたデザインとなっています。


![sample_theme_E](https://user-images.githubusercontent.com/59504885/236095986-5a0228f2-6c70-4a22-a51e-d974b2c146e1.jpg)

### theme_U

内田研究室の先輩が使っていたPower Pointのデザインを参考に作成したデザインです。主にゼミの発表などの際に使用しています。

![sample_theme_U](https://user-images.githubusercontent.com/59504885/236095998-00c07963-139f-49a5-a4c0-716d67740a73.jpg)

## 使い方・利用例
基本的な使い方は`samples/sample.md`を参照してください。
また、参考用に実際に輪講で作成した際のMarkdownファイルが`samples`にあります。
