from core.proxies.main_proxy import MainProxy


def main():
    print('--------------------- start -------------------------')
    main_proxy = MainProxy(predict_mode=True)
    main_proxy.run()


if __name__ == '__main__':
    main()
