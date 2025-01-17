from manim import *

class Hello(Scene):
    def construct(self):
        text = Tex("hello")
        self.play(Write(text))
        self.wait(3)

        t = Tex(r"$x_{1, 2} = \frac{-b \pm \sqrt{D}}{2a}$", font_size=96)
        self.add(t)
        self.wait(3)

class Shapes(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        self.play(Create(circle))
        self.play(circle.animate.shift(LEFT))
        self.play(Create(square))
        self.play(square.animate.shift(RIGHT))
        self.wait()