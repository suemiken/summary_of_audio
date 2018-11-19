# audio_summary
フィードバック型のニューラルネットワークを用いて音声の要約を実現する。
TF-IDF法による要約に加えて、発話者の前後の感情を考慮した要約を目指す。

# Dependency
Dependency is written in the requrements.txt concerning python.
It is possible to change audio into text by using IBM watson speech to text

IBM watson speech to textを使って音声を文字変換しています。
pythonのライブラリはrequrements.txtに記載。

# Setup
I used conda to install python library.
I advise to use it because of  becoming ease to manage.

condaを使ってライブラリを管理しています。
楽にバージョン管理ができるためお勧めです。

# Usage
create_coporaでは音声データを文字に変換する処理するプログラムが入っている。
corporaには音声データ→jsonファイル→文字に変換し、感情を人手で付加したタグ付きコーパスが入っている。
summary_partには要約する際に必要なプログラムが入っている。


# References
Get To The Point: Summarization with Pointer-Generator Networks
