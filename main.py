import gradio as gr
from app.entities.entity import *
from app.presenters.presenter import *
from app.views.view import *
from app.interactors.interactor import *
from app.routers.router import *


def main():
    demo.launch(share=True)


if __name__ == "__main__":
    main()
