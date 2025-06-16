function generateUUID(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.floor(Math.random() * 16);
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Получение или генерация токена и сохранение его в localStorage
function getOrCreateToken(): string {
    let token = localStorage.getItem('sessionToken');
    if (!token) {
        token = generateUUID();
        localStorage.setItem('sessionToken', token);
    }
    return token;
}

const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";

document.addEventListener("DOMContentLoaded", () => {
    const roomSelection = document.getElementById('room-selection') as HTMLDivElement;
    const lobbyPanel = document.getElementById('lobby-panel') as HTMLDivElement;
    const playersPanel = document.getElementById('players-panel') as HTMLDivElement;
    const createRoomBtn = document.getElementById('create-room-btn') as HTMLButtonElement;
    const joinRoomBtn = document.getElementById('join-room-btn') as HTMLButtonElement;
    const leaveRoomBtn = document.getElementById('leave-room-btn') as HTMLButtonElement;
    const roomInfo = document.getElementById('room-info') as HTMLSpanElement;
    const chatWindow = document.getElementById('chat-window') as HTMLDivElement;
    const chatInput = document.getElementById('chat-input') as HTMLInputElement;
    const sendMsgBtn = document.getElementById('send-msg-btn') as HTMLButtonElement;
    const readyBtn = document.getElementById('ready-btn') as HTMLButtonElement;
    const roomCodeInput = document.getElementById('room-code') as HTMLInputElement;
    const playerNameInput = document.getElementById('player-name') as HTMLInputElement;
    const playersList = document.getElementById('players-list') as HTMLDivElement;

    let isReady = false;
    let currentRoom = '';
    let username: string;
    let ws: WebSocket | null = null;

    let token = getOrCreateToken();
    if (!token) {
        token = generateUUID();
    }

    // Обновления списка игроков
    function updatePlayersList(players: { name: string; ready: boolean }[]) {
        playersList.innerHTML = '';
        players.forEach(player => {
            const playerDiv = document.createElement('div');
            playerDiv.textContent = `${player.name} — ${player.ready ? 'Готов' : 'Не готов'}`;
            playersList.appendChild(playerDiv);
        });
    }

    // Добавление сообщения в чат
    function appendMessage(sender: string, message: string) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message');
        msgDiv.textContent = sender ? `${sender}: ${message}` : message;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    async function showLobby(roomId: string, playerName: string, isImplicit: boolean = false) {
        // Отправляем запрос на подключение к комнате с указанием имени
        try {
            const joinRes = await fetch(
                `/join_room?room_id=${roomId}&token=${token}&name=${encodeURIComponent(playerName)}`,
                { method: "POST" }
            );
            if (!joinRes.ok) {
                let errorMessage = "";
                try {
                    const errorData = await joinRes.json();
                    if (errorData.detail) {
                        errorMessage = `Ошибка: ${errorData.detail}`;
                    } else {
                        errorMessage = `Ошибка (без detail): ${JSON.stringify(errorData)}`;
                    }
                } catch {
                    // Если не JSON, читаем как текст
                    const errorText = await joinRes.text();
                    errorMessage = `Ошибка (текст): ${errorText}`;
                }
                throw new Error(errorMessage);
            }
            currentRoom = roomId;

            localStorage.setItem("currentRoom", roomId);
            localStorage.setItem("playerName", playerName);
            roomInfo.textContent = `Комната: ${roomId}`;
            roomSelection.classList.add('hidden');
            lobbyPanel.classList.remove('hidden');
            playersPanel.classList.remove('hidden');
        } catch (error) {
            localStorage.removeItem("currentRoom");
            console.error("Ошибка запроса join_room:", error);

            // Показываем алерт только при явном подключении
            if (!isImplicit) {
                alert(error.message || "Неизвестная ошибка подключения к комнате");
            }
            return;
        }

        appendMessage("Система", `Вы подключились к комнате ${roomId}`);

        // Устанавливаем WebSocket-соединение
        ws = new WebSocket(`${WS_PROTOCOL}://${window.location.host}/ws/lobby/${roomId}/${token}`);

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "players_update") {
                    updatePlayersList(data.players);
                } else if (data.type === "game_start") {
                    // 1) Закрываем лобби-сокет:
                    ws.close();

                    // 2) Переходим на страницу игры:
                    window.location.href = `/game/${roomId}?token=${token}&username=${playerName}&room_id=${roomId}`;
                }
            } catch (e) {
                // Если не JSON, считаем как обычное текстовое сообщение
                appendMessage("", event.data);
            }
        };

        ws.onclose = () => {
            appendMessage("Система", "Соединение закрыто");
        };

        ws.onerror = (error) => {
            console.error("WebSocket ошибка", error);
        };
    }

    // Выход из комнаты
    function leaveRoom() {
        currentRoom = '';
        localStorage.removeItem("currentRoom");
        localStorage.removeItem("playerName");
        if (ws) {
            ws.close();
            ws = null;
        }
        lobbyPanel.classList.add('hidden');
        playersPanel.classList.add('hidden');
        roomSelection.classList.remove('hidden');
        chatWindow.innerHTML = '';
        playersList.innerHTML = '';
    }

    // Создание комнаты
    createRoomBtn.addEventListener('click', async () => {
        const playerName = playerNameInput.value.trim();
        if (!playerName) {
            alert("Введите имя игрока");
            return;
        }
        try {
            const res = await fetch(`/create_room`, { method: "POST" });
            const data = await res.json();
            const roomId = data.room_id;
            showLobby(roomId, playerName);
        } catch (error) {
            console.error("Ошибка создания комнаты", error);
            alert("Не удалось создать комнату");
        }
    });

    // Подключение к комнате по коду
    joinRoomBtn.addEventListener('click', async () => {
        const roomId = roomCodeInput.value.trim();
        const playerName = playerNameInput.value.trim();
        if (!playerName) {
            alert("Введите имя игрока");
            return;
        }
        if (roomId !== '') {
            showLobby(roomId, playerName);
        } else {
            alert("Введите код комнаты");
        }
    });

    // Выход из комнаты
    leaveRoomBtn.addEventListener('click', () => {
        leaveRoom();
    });

    // Отправка сообщения в чат
    sendMsgBtn.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message !== '' && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(message);
            chatInput.value = '';
        }
    });

    // Отправка сообщения при нажатии Enter
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMsgBtn.click();
        }
    });

    // Переключение состояния готовности
    readyBtn.addEventListener('click', () => {
        isReady = !isReady;
        readyBtn.textContent = isReady ? "Не готов" : "Готов";
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(isReady ? "READY" : "NOT_READY");
        }
        appendMessage("Система", isReady ? "Вы готовы к игре!" : "Вы отменили готовность");
    });

    // При загрузке страницы проверяем, не сохранилась ли информация о комнате и имени
    // Например при обновлении страницы, или после сетевого сбоя возвращаемся обратно в лобби
    const savedRoom = localStorage.getItem("currentRoom");
    const savedName = localStorage.getItem("playerName");
    if (savedRoom && savedName) {
        showLobby(savedRoom, savedName, true); // добавляем true для неявного подключения
    }
});
