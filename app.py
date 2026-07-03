"""
CineSmart IA — Chatbot de atención al cliente de Cineplanet
Software independiente (Streamlit) que usa el modelo K-Means ya entrenado
en el notebook para segmentar al cliente y personalizar las respuestas
de un asistente conversacional con Gemini (Google).

CÓMO CORRERLO:
1. pip install -r requirements.txt
2. Asegúrate de tener 'modelo_cineplanet.pkl' en la misma carpeta que este archivo
   (lo exportas desde la última celda del notebook de Colab).
3. streamlit run app.py
4. Se abre automáticamente en tu navegador (normalmente http://localhost:8501)
"""

import streamlit as st
import pandas as pd
import joblib
from google import genai
from google.genai import types

st.set_page_config(page_title="CineSmart IA — Cineplanet", page_icon="🎬", layout="centered")

# --------------------------------------------------------------------------
# Carga del modelo entrenado (exportado desde el notebook)
# --------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_cineplanet.pkl")

try:
    modelo = cargar_modelo()
    kmeans = modelo["kmeans"]
    scaler = modelo["scaler"]
    segmentos = modelo["segmentos"]
    campanias = modelo["campanias"]
except FileNotFoundError:
    st.error(
        "No se encontró 'modelo_cineplanet.pkl'. Expórtalo desde la última celda "
        "del notebook de Colab y colócalo en la misma carpeta que app.py."
    )
    st.stop()

# --------------------------------------------------------------------------
# Estado de la sesión
# --------------------------------------------------------------------------
if "cliente_actual" not in st.session_state:
    st.session_state.cliente_actual = None
if "chat" not in st.session_state:
    st.session_state.chat = None
if "historial_ui" not in st.session_state:
    st.session_state.historial_ui = []  # solo para mostrar en pantalla
if "client" not in st.session_state:
    st.session_state.client = None


def construir_contexto_sistema(segmento_cliente=None, campanias=None):
    base = """Eres el asistente virtual de atención al cliente de Cineplanet, una cadena de cines en Perú.
Ayudas con: promociones, horarios de funciones, cartelera, ubicación de sedes, reservas de entradas,
cancelaciones/reembolsos, quejas y la membresía Cineplanet Play.

Reglas:
- Responde siempre en español, de forma breve, cordial y profesional (máximo 3-4 líneas).
- Cuando tengas datos concretos (como la promoción y el combo del cliente, más abajo), MENCIÓNALOS
  directamente y con sus condiciones. NO remitas al cliente a la app/web si ya tienes el dato.
- Solo remite a la app o web oficial para cosas que realmente no tienes (ej. horario exacto de una
  función puntual, disponibilidad de asientos en tiempo real, sedes exactas).
- Si el cliente muestra una queja o reclamo, responde con empatía y ofrece derivarlo con un asesor humano.
"""
    if segmento_cliente:
        campania = (campanias or {}).get(segmento_cliente['nombre'], {})
        promocion = campania.get('Promoción', 'No hay una promoción específica registrada para este perfil.')
        combo = campania.get('Combo', 'No hay un combo específico registrado para este perfil.')

        base += f"""
Información del cliente actual, para personalizar tus respuestas de forma natural
(sin mencionar explícitamente palabras como "segmento" o "cluster" al cliente):
- Perfil de cliente: {segmento_cliente['nombre']}
- Promoción vigente para este cliente: {promocion}
- Combo recomendado para este cliente: {combo}
- Estrategia de marketing para este perfil: {segmento_cliente['estrategia']}
- Canal de comunicación preferido: {segmento_cliente['canal']}

Cuando el cliente pregunte por promociones, descuentos o beneficios, ofrécele DIRECTAMENTE la
promoción y el combo indicados arriba, como si fueran datos reales y vigentes.
"""
    return base


def iniciar_chat():
    config = types.GenerateContentConfig(
        system_instruction=construir_contexto_sistema(st.session_state.cliente_actual, campanias)
    )
    st.session_state.chat = st.session_state.client.chats.create(
        model="gemini-2.5-flash", config=config
    )
    st.session_state.historial_ui = []


# --------------------------------------------------------------------------
# Barra lateral: API key + identificación del cliente
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    # Intenta tomar la key desde los "Secrets" de Streamlit Cloud primero.
    # Si no existe (ej. corriendo localmente sin configurarla), pide el campo manual.
    api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None

    if api_key:
        if st.session_state.client is None:
            st.session_state.client = genai.Client(api_key=api_key)
        st.success("Conectado a Gemini ✅")
    else:
        api_key_manual = st.text_input("API key de Gemini", type="password")
        if api_key_manual and st.session_state.client is None:
            st.session_state.client = genai.Client(api_key=api_key_manual)
            st.success("Conectado a Gemini ✅")

    st.divider()
    st.header("👤 Identificar cliente")
    st.caption("Estos datos alimentan el modelo K-Means para detectar el segmento del cliente.")

    edad = st.number_input("Edad", min_value=10, max_value=90, value=25)
    ingreso = st.number_input("Ingreso mensual (S/)", min_value=0, value=2000, step=100)
    frecuencia = st.number_input("Frecuencia de visitas al mes", min_value=0, value=2)
    precio = st.number_input("Precio de entrada habitual (S/)", min_value=0, value=18)

    if st.button("Identificar segmento", use_container_width=True):
        if st.session_state.client is None:
            st.warning("Primero ingresa tu API key de Gemini.")
        else:
            datos = pd.DataFrame({
                "Edad": [edad],
                "Ingreso_Mensual": [ingreso],
                "Frecuencia_Visitas_Mes": [frecuencia],
                "Precio_Entrada": [precio],
            })
            datos_scaled = scaler.transform(datos)
            cluster = kmeans.predict(datos_scaled)[0]
            st.session_state.cliente_actual = segmentos[cluster]
            iniciar_chat()
            st.success(f"Segmento detectado: {st.session_state.cliente_actual['nombre']}")

    if st.session_state.cliente_actual:
        st.info(f"**Segmento actual:** {st.session_state.cliente_actual['nombre']}")

# --------------------------------------------------------------------------
# Cuerpo principal: interfaz de chat
# --------------------------------------------------------------------------
st.title("🎬 CineSmart IA")
st.caption("Asistente virtual de atención al cliente — Cineplanet")

if st.session_state.client is None:
    st.warning("Ingresa tu API key de Gemini en la barra lateral para comenzar.")
    st.stop()

if st.session_state.chat is None:
    iniciar_chat()

# Mostrar historial de la conversación
for rol, mensaje in st.session_state.historial_ui:
    with st.chat_message(rol):
        st.markdown(mensaje)

# Entrada de mensaje del usuario
mensaje_usuario = st.chat_input("Escribe tu mensaje...")
if mensaje_usuario:
    st.session_state.historial_ui.append(("user", mensaje_usuario))
    with st.chat_message("user"):
        st.markdown(mensaje_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                respuesta = st.session_state.chat.send_message(mensaje_usuario)
                texto_respuesta = respuesta.text
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    texto_respuesta = (
                        "⚠️ Se alcanzó el límite de solicitudes gratuitas de la API por ahora. "
                        "Espera un minuto y vuelve a intentar."
                    )
                else:
                    texto_respuesta = f"⚠️ Ocurrió un error al conectar con el asistente: {e}"
            st.markdown(texto_respuesta)
    st.session_state.historial_ui.append(("assistant", texto_respuesta))
