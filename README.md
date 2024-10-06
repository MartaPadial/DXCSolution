

![DXC Logo](https://dxc.com/content/dam/dxc/projects/dxc-com/us/images/about-us/newsroom/logos-for-media/horizontal/DXC%20Logo%20Horiz_Purple+Black%20RGB.png)

## Prueba Técnica: Chatbot Personalizado con Langchain y GPT-3 ##

### 📝 Descripción

Esta es la prueba tecnica para evaluar las capacidades técnicas de 👏 Marta Padial 👏 orientadas a AI y que contendrá el código fuente para un chatbot personalizado desarrollado utilizando la biblioteca Langchain y un modelo de lenguaje GPT-3 o cualquier otra alternativa de modelo. El objetivo principal de este proyecto es crear un agente conversacional capaz de responder a consultas sobre cualquier dominio segun el modelo elegido.


### ⭐ Funcionalidades requeridas

* Introducción de Prompts: Bien sea mediante el uso de un IU o bien directamente desde consola.

* Uso de langchain: Facilita la interacion de un usuario contra modelos. 

* Conexión con un modelo: Integración con un modelo de lenguaje para generar respuestas coherentes y contextualmente relevantes. Se recomienda GPT-3 aunque se podria usar cualquiera.

* Realizar flujos de Diálogo Complejos: Soporte para diálogos multi-turno y ramificaciones en función de las entradas del usuario.

### 📚 Estructura del Proyecto

1. models: Contiene los modelos de lenguaje utilizados (por ejemplo, archivos de configuración de GPT-3 o del modelo usado.).

2. data: Almacena los datos de entrenamiento o configuración del chatbot (intenciones, entidades, etc.).

3. utils: Incluye funciones utilitarias para el preprocesamiento de texto, la gestión de conversaciones y otras tareas.

4. chatbot.py: Contiene la implementación principal del chatbot, incluyendo la lógica de conversación y la interacción con el usuario.

### 📃 Requisitos

* Python (versión recomendada: 3.x)

* Bibliotecas: Langchain, OpenAI (o la API de tu proveedor de modelos de lenguaje), y otras dependencias listadas en requirements.txt.

requirements.txt

```
langchain==0.0.15
openai==0.27.2
```

### 🛠️ Herramientas

No se impone el uso de ninguna herramienta, ni formato de presentacion. Pero quizas puedan resultar interesantes las siguientes herramientas en funcion de como se decida abordar los problemas:

* Plataforma integral de desarrollo: https://colab.research.google.com/
* Interface de usuario: https://streamlit.io/ 
* Hub de modelos disponibles: https://huggingface.co/
* Cualquier herramienta de generacion de codigo es valida


Las herramientas que seguro que hay que usar son:

* Documentación de Langchain: https://langchain.readthedocs.io/
* Documentación de OpenAI: https://beta.openai.com/docs/


### ⚙️ Instalación

Clona este repositorio:

**Bash**

```
git clone https://github.com/tu-usuario/tu-repositorio.git
```


Crea un entorno virtual y activa:

**Bash**

```
python -m venv venv

source venv/bin/activate
```

Instala las dependencias:

**Bash**
```
pip install -r requirements.txt
```


Configura las credenciales de tu API de OpenAI.

### 🗨️ Uso

Para ejecutar el chatbot, ejecuta el siguiente comando desde la línea de comandos:

**Bash**
```
python chatbot.py
```

### 📦 Despliegue 
Se puede optar por la solucion de despliegue que se crea conveniente, bien en local y distribuir un video de ejecucion, o bien desplegarlo en soluciones de contenerizacion de cloud publica como:

* https://www.pythonanywhere.com
* https://www.heroku.com/


### 🚀 What if...?

¿Como minimizarias el consumo de tokens?

¿Como harias mas rapidas las consultas contra el modelo?

### 🤝 Gracias

Antes de nada agradecerte el tiempo dedicado. Se asume que es tiempo libre el que estas dedicando para esta prueba, por lo que apoyate en el recurso que necesites, desde nosostros hasta cualquier generador de codigo que abierto que veas. Recuerda que la dedicacion para conseguir esto seria unas 3-4 horas.# DXCSolution
# DXCSolution
