import faulthandler; faulthandler.enable()
import tkinter
print('tkinter TK version:', tkinter.TkVersion)
import PySimpleGUI as sg
print('PySimpleGUI version:', sg.version)
w = sg.Window('probe', [[sg.Text('hello')]], finalize=True)
e, v = w.read(timeout=1500)
w.close()
print('window read event:', repr(e))
print('PROBE OK')
