import mitsuba as mi

def main():
    mi.set_variant('scalar_rgb')
    _ = mi.Sensor
    print(f'Mitsuba version: {mi.__version__} installed successfully!')

if __name__ == '__main__':
    main()
