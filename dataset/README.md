Gerando texto com Ollama:

```
$ wget http://localhost:11434/api/generate --quiet -O- --post-data="{\"model\": \"llama3.1\", \"prompt\": \"MY PROMPT\", \"stream\": false}" | jq --raw-output ".response"
```

Gerando 100 poemas:

```
$ for i in $(seq 1 100); do echo -ne "\r$i" && echo -e "`./llama3.1 escreva um poema curto em português do Brasil, sem título, sem enunciado, e com linguajar simples`\n@ >> poemas.txt; done
```

Contando a quantidade de ocorrências (pelo terminador "@"):

```
$ grep -c "^@$" poemas-10k.txt
10000
$ grep -c "^@$" poemas-100k.txt
100000
```

Removendo linhas que contenham "Aqui vai\*:" e "Note\*:":

```
$ sed -i '/[Aa]qui vai.*:/,+1d' poemas-100000.txt 
$ sed -i '/[N]ote.*:/d' poemas-100000.txt 
```
