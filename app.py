"""Venice Travel Assistant: Firebase sign-in and session-only RAG chat."""
import streamlit as st
from firebase_client import FirebaseError, create_account, sign_in, require_session, validate_config
from rag import RagError, answer_question
from config import MIN_SIMILARITY


def forget_user():
    # Only visitor state is cleared; the shared public-guide embedding cache remains.
    st.session_state.clear()


def show_sources(sources):
    with st.expander('Retrieved guide entries and cosine scores'):
        st.caption('Similarity is not a probability. Entries below the cutoff are not sent to the model.')
        for source in sources:
            used = source['score'] >= MIN_SIMILARITY
            st.markdown(f"**[{source['id']}] {source['title']}**")
            st.caption(f"Cosine: {source['score']:.4f} · {'Included in context' if used else 'Below cutoff'}")
            st.write(source['text'])
            st.link_button('Open official source', source['source'])


st.set_page_config(page_title='Venice Travel Assistant', page_icon='🛶')
st.title('Venice Travel Assistant')
st.caption('Explore Venice with answers grounded in our 18-entry travel guide.')

if notice := st.session_state.pop('notice', None):
    st.info(notice)
try:
    validate_config()
except FirebaseError as error:
    st.info(str(error))
    st.stop()

if 'user' not in st.session_state:
    st.subheader('Sign in to chat')
    with st.form('account_form', clear_on_submit=True):
        email = st.text_input('Email')
        password = st.text_input('Password', type='password')
        login = st.form_submit_button('Sign in')
        register = st.form_submit_button('Create account')
    if login or register:
        try:
            if register:
                create_account(email, password)
                st.success('Account created. Sign in with your email and password.')
            else:
                user = sign_in(email, password)
                forget_user()
                st.session_state['user'] = user
                st.rerun()
        except FirebaseError as error:
            st.error(str(error))
    st.stop()

with st.sidebar:
    st.write(f"Signed in as {st.session_state['user']['email']}")
    if st.button('Sign out'):
        forget_user()
        st.rerun()
    st.caption('Chat history is kept only for this session and is cleared at sign-out.')

try:
    require_session(st.session_state['user'])
except FirebaseError as error:
    forget_user()
    st.session_state['notice'] = str(error)
    st.rerun()

messages = st.session_state.setdefault('messages', [])
if not messages:
    st.info('Try: What can I see at the Murano Glass Museum? Please name the place in each question.')
for message in messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            show_sources(message['sources'])

question = st.chat_input('Ask about Venice', max_chars=2000)
if question is not None:
    if not question.strip():
        st.warning('Please enter a question.')
        st.stop()
    try:
        require_session(st.session_state['user'])
        with st.spinner('Searching the guide and preparing an answer…'):
            result = answer_question(question)
        # Do not publish an answer if the session expired during a slow request.
        require_session(st.session_state['user'])
    except FirebaseError as error:
        forget_user()
        st.session_state['notice'] = str(error)
        st.rerun()
    except RagError as error:
        st.error(str(error))
        st.caption('Your previous chat is unchanged. You can submit the question again.')
    else:
        messages.extend([
            {'role': 'user', 'content': question.strip()},
            {'role': 'assistant', 'content': result['answer'], 'sources': result['sources']},
        ])
        st.rerun()
